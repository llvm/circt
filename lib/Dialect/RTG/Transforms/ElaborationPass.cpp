//===- ElaborationPass.cpp - RTG ElaborationPass implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass elaborates the random parts of the RTG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGVisitors.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/IRMapping.h>
#include <random>

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_ELABORATIONPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

#define DEBUG_TYPE "rtg-elaboration"

static unsigned interpret(Value value) { return 1; }

namespace {
class ElaborationPass : public rtg::impl::ElaborationPassBase<ElaborationPass> {
public:
  ElaborationPass() : ElaborationPassBase() {}
  ElaborationPass(const ElaborationPass &other) : ElaborationPassBase(other) {}
  ElaborationPass(const rtg::ElaborationOptions &options)
      : ElaborationPassBase(), options(options) {
    if (options.seed.has_value())
      seed = options.seed.value();
  }
  void runOnOperation() override;
  LogicalResult runOnTest(TestOp testOp);
  void cloneTargetsIntoTests();

private:
  ElaborationOptions options;
  Pass::Option<unsigned> seed{*this, "seed",
                              llvm::cl::desc("Seed for the RNG.")};
};

class Elaborator : public RTGOpVisitor<Elaborator, LogicalResult> {
public:
  using EvalValue = std::variant<size_t, DenseSet<size_t>>;
  using Base = RTGOpVisitor<Elaborator, LogicalResult>;
  using Base::visitOp;

  Elaborator(const ElaborationOptions &options) {
    std::random_device dev;
    unsigned s = dev();
    if (options.seed.has_value())
      s = *options.seed;
    rng = std::mt19937(s);
  }

  LogicalResult visitUnhandledOp(Operation *op) {
    return op->emitError("not supported");
  }

  LogicalResult visitResourceOp(ResourceOpInterface op) {
    return op->emitError("resource op not supported");
  }

  LogicalResult visitInstruction(InstructionOpInterface op) {
    return success();
  }

  LogicalResult visitOp(SelectRandomOp op) {
    std::vector<float> idx;
    std::vector<float> prob;
    idx.push_back(0);
    for (unsigned i = 1; i <= op.getRatios().size(); ++i) {
      idx.push_back(i);
      unsigned p = interpret(op.getRatios()[i - 1]);
      prob.push_back(p);
    }

    std::piecewise_constant_distribution<float> dist(idx.begin(), idx.end(),
                                                     prob.begin());

    unsigned selected = dist(rng);

    auto sequence =
        op.getSequences()[selected].getDefiningOp<rtg::SequenceOp>();
    if (!sequence)
      return failure();

    IRRewriter rewriter(op);
    auto *clone = sequence->clone();
    rewriter.inlineBlockBefore(&clone->getRegion(0).front(), op,
                               op.getSequenceArgs()[selected]);
    clone->erase();

    op->erase();
    return success();
  }

  LogicalResult visitOp(SetCreateOp op) {
    if (!isa<ContextResourceType>(op.getSet().getType().getElementType()))
      return failure();

    DenseSet<size_t> set;
    for (auto val : op.getElements()) {
      assert(results.contains(val));
      set.insert(std::get<size_t>(results[val]));
    }

    results[op.getSet()] = set;
    return success();
  }

  LogicalResult visitOp(SetSelectRandomOp op) {
    if (!isa<ContextResourceType>(op.getSet().getType().getElementType()))
      return failure();

    assert(results.contains(op.getSet()));

    auto set = std::get<DenseSet<size_t>>(results[op.getSet()]);
    SmallVector<size_t> setVec(set.begin(), set.end());

    std::uniform_int_distribution<size_t> dist(0, setVec.size() - 1);
    size_t selected = dist(rng);

    results[op.getResult()] = setVec[selected];
    return success();
  }

  LogicalResult visitOp(SetDifferenceOp op) {
    if (!isa<ContextResourceType>(op.getOriginal().getType().getElementType()))
      return failure();

    assert(results.contains(op.getOriginal()) &&
           results.contains(op.getDiff()));

    auto original = std::get<DenseSet<size_t>>(results[op.getOriginal()]);
    auto diff = std::get<DenseSet<size_t>>(results[op.getDiff()]);

    DenseSet<size_t> result;
    std::set_difference(original.begin(), original.end(), diff.begin(),
                        diff.end(), result.end());

    results[op.getResult()] = result;
    return success();
  }

  LogicalResult visitOp(OnContextOp op) {
    assert(results.contains(op.getContexts()));

    SmallVector<int64_t> selectors;
    if (auto setTy = dyn_cast<SetType>(op.getContexts().getType())) {
      auto set = std::get<DenseSet<size_t>>(results[op.getContexts()]);
      selectors = {set.begin(), set.end()};
      llvm::sort(selectors);
    } else {
      selectors.push_back(std::get<size_t>(results[op.getContexts()]));
    }

    OpBuilder builder(op);
    auto renderedContextOp = builder.create<RenderedContextOp>(
        op.getLoc(), selectors, selectors.size());
    IRMapping mapper;
    for (unsigned i = 0; i < selectors.size(); ++i)
      op.getBodyRegion().cloneInto(&renderedContextOp->getRegion(i), mapper);

    op->erase();
    return success();
  }

  LogicalResult visitExternalOp(Operation *op) {
    return op->emitError("external op not supported");
  }

  LogicalResult visitContextResourceOp(ContextResourceOpInterface op) {
    // FIXME: don't just assume result#0
    results[op->getResult(0)] = op.getResourceId();
    return success();
  }

  LogicalResult elaborate(TestOp testOp) {
    DenseSet<Operation *> visited;
    SmallVector<Operation *> worklist;
    for (auto &op : *testOp.getBody())
      if (op.getNumResults() == 0)
        worklist.push_back(&op);

    // FIXME: visit nested regions as well after their parent region was already
    // elaborated

    while (!worklist.empty()) {
      auto *curr = worklist.back();
      // if (llvm::all_of(curr->getResults(), [&](auto val) { return
      // results.contains(val); })) {
      //   worklist.pop_back();
      //   continue;
      // }

      bool addedSomething = false;
      for (auto val : curr->getOperands()) {
        if (results.contains(val))
          continue;

        auto *defOp = val.getDefiningOp();
        if (!visited.contains(defOp)) {
          visited.insert(defOp);
          worklist.push_back(defOp);
          addedSomething = true;
        }
      }

      if (addedSomething)
        continue;

      if (failed(dispatchOpVisitor(curr)))
        return failure();

      worklist.pop_back();
    }

    return success();
  }

private:
  std::mt19937 rng;
  DenseMap<Value, EvalValue> results;
};
} // end namespace

void ElaborationPass::runOnOperation() {
  cloneTargetsIntoTests();

  auto moduleOp = getOperation();
  for (auto testOp : moduleOp.getOps<TestOp>())
    if (failed(runOnTest(testOp)))
      return signalPassFailure();
}

void ElaborationPass::cloneTargetsIntoTests() {
  for (auto target :
       llvm::make_early_inc_range(getOperation().getOps<TargetOp>())) {
    for (auto test : getOperation().getOps<TestOp>()) {
      // FIXME: we need to clone the test and make sure there's a copy of it for
      // every matching target
      if (test.getTargetType().getEntryTypes().empty())
        continue;
      // TODO: allow target refinements, just not coarsening
      IRRewriter rewriter(test);
      if (target.getTargetType() == test.getTargetType()) {
        rewriter.setInsertionPoint(target);
        auto newTarget = cast<TargetOp>(rewriter.clone(*target));
        auto *yield = newTarget.getBody()->getTerminator();
        rewriter.inlineBlockBefore(newTarget.getBody(), test.getBody(),
                                   test.getBody()->begin());
        rewriter.replaceAllUsesWith(test.getBody()->getArguments(),
                                    yield->getOperands());
        test.getBody()->eraseArguments(0, test.getBody()->getNumArguments());
        rewriter.eraseOp(yield);
        rewriter.eraseOp(newTarget);
      }
      test.setTargetType(TargetType::get(&getContext(), {}, {}));
    }
    target->erase();
  }
}

LogicalResult ElaborationPass::runOnTest(TestOp testOp) {
  return Elaborator(options).elaborate(testOp);
}

std::unique_ptr<Pass>
rtg::createElaborationPass(const rtg::ElaborationOptions &options) {
  return std::make_unique<ElaborationPass>(options);
}
