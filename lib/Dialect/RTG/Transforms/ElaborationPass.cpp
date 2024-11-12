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

#include "circt/Dialect/RTG/IR/ArithVisitors.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGVisitors.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/IR/IRMapping.h>
#include <random>
#include <variant>

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

struct SequenceClosure {
  StringAttr symbol;
  SmallVector<Value> args;
};

class Elaborator : public RTGOpVisitor<Elaborator, LogicalResult>,
                   public arith::ArithOpVisitor<Elaborator, LogicalResult> {
public:
  using LeafEvalValue = std::variant<size_t, SequenceClosure>;
  using EvalValue = std::variant<SmallVector<LeafEvalValue>,
                                 SmallVector<std::pair<LeafEvalValue, size_t>>,
                                 LeafEvalValue>;
  using RTGOpVisitor<Elaborator, LogicalResult>::visitOp;
  using ArithOpVisitor<Elaborator, LogicalResult>::visitOp;

  Elaborator(SymbolTable &table, const ElaborationOptions &options)
      : symTable(table) {
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

  LogicalResult visitOp(arith::ConstantOp op) {
    auto attr = dyn_cast<IntegerAttr>(op.getValue());
    if (!attr)
      return failure();

    results[op.getResult()] =
        std::make_unique<EvalValue>(attr.getValue().getZExtValue());
    return success();
  }

  LogicalResult visitOp(SequenceClosureOp op) {
    // TODO: ignored for now because it's handled by SelectRandomOp
    SequenceClosure closure;
    closure.symbol = op.getSequenceAttr();
    closure.args = op.getArgs();
    results[op.getResult()] = std::make_unique<EvalValue>(closure);
    return success();
  }

  LogicalResult visitOp(BagCreateOp op) {
    // if (!isa<ContextResourceType>(op.getBag().getType().getElementType()))
    //   return failure();

    SmallVector<std::pair<LeafEvalValue, size_t>> bag;
    for (auto [val, weight] : llvm::zip(op.getElements(), op.getWeights())) {
      assert(results.contains(val));
      assert(results.contains(weight));
      bag.emplace_back(
          std::get<LeafEvalValue>(*results[val]),
          std::get<size_t>(std::get<LeafEvalValue>(*results[weight])));
    }

    results[op.getBag()] = std::make_unique<EvalValue>(bag);
    return success();
  }

  LogicalResult visitOp(InvokeSequenceOp op) {
    auto sequenceClosure = std::get<SequenceClosure>(
        std::get<LeafEvalValue>(*results[op.getSequence()]));

    IRRewriter rewriter(op);
    auto sequence = symTable.lookupNearestSymbolFrom<SequenceOp>(
        op->getParentOfType<ModuleOp>(), sequenceClosure.symbol);
    auto *clone = sequence->clone();
    rewriter.inlineBlockBefore(&clone->getRegion(0).front(), op,
                               sequenceClosure.args);
    clone->erase();

    op->erase();
    return success();
  }

  LogicalResult visitOp(BagSelectRandomOp op) {
    assert(results.contains(op.getBag()));

    auto bag = std::get<SmallVector<std::pair<LeafEvalValue, size_t>>>(
        *results[op.getBag()]);
    SmallVector<LeafEvalValue> bagVec;
    SmallVector<double> weights, idx;

    llvm::errs() << "Size: " << bag.size() << "\n";
    for (auto [i, elAndWeight] : llvm::enumerate(bag)) {
      auto [el, weight] = elAndWeight;
      bagVec.push_back(el);
      weights.push_back(weight);
      idx.push_back(i);
    }

    std::piecewise_constant_distribution<double> dist(idx.begin(), idx.end(),
                                                      weights.begin());
    size_t selected = std::round(dist(rng));

    results[op.getResult()] = std::make_unique<EvalValue>(bagVec[selected]);
    return success();
  }

  // LogicalResult visitOp(BagDifferenceOp op) {
  //   if
  //   (!isa<ContextResourceType>(op.getOriginal().getType().getElementType()))
  //     return failure();

  //   assert(results.contains(op.getOriginal()) &&
  //          results.contains(op.getDiff()));

  //   auto original = std::get<SmallVector<std::pair<LeafEvalValue,
  //   size_t>>>(*results[op.getOriginal()]); auto diff =
  //   std::get<SmallVector<std::pair<LeafEvalValue,
  //   size_t>>>(*results[op.getDiff()]);

  //   SmallVector<std::pair<LeafEvalValue, size_t>> result;
  //   // FIXME: do proper bag difference
  //   std::set_difference(original.begin(), original.end(), diff.begin(),
  //                       diff.end(), result.end());

  //   results[op.getResult()] = std::make_unique<EvalValue>(result);
  //   return success();
  // }

  LogicalResult visitOp(SetCreateOp op) {
    if (!isa<ContextResourceType>(op.getSet().getType().getElementType()))
      return failure();

    SmallVector<LeafEvalValue> set;
    for (auto val : op.getElements()) {
      assert(results.contains(val));
      set.emplace_back(std::get<LeafEvalValue>(*results[val]));
    }

    results[op.getSet()] = std::make_unique<EvalValue>(set);
    return success();
  }

  LogicalResult visitOp(SetSelectRandomOp op) {
    if (!isa<ContextResourceType>(op.getSet().getType().getElementType()))
      return failure();

    assert(results.contains(op.getSet()));

    auto set = std::get<SmallVector<LeafEvalValue>>(*results[op.getSet()]);
    SmallVector<LeafEvalValue> setVec(set.begin(), set.end());

    std::uniform_int_distribution<size_t> dist(0, setVec.size() - 1);
    size_t selected = dist(rng);

    results[op.getResult()] = std::make_unique<EvalValue>(setVec[selected]);
    return success();
  }

  // LogicalResult visitOp(SetDifferenceOp op) {
  //   if
  //   (!isa<ContextResourceType>(op.getOriginal().getType().getElementType()))
  //     return failure();

  //   assert(results.contains(op.getOriginal()) &&
  //          results.contains(op.getDiff()));

  //   auto original =
  //   std::get<SmallVector<LeafEvalValue>>(*results[op.getOriginal()]); auto
  //   diff = std::get<SmallVector<LeafEvalValue>>(*results[op.getDiff()]);

  //   SmallVector<LeafEvalValue> result;
  //   std::set_difference(original.begin(), original.end(), diff.begin(),
  //                       diff.end(), result.end());

  //   results[op.getResult()] = std::make_unique<EvalValue>(result);
  //   return success();
  // }

  LogicalResult visitOp(OnContextOp op) {
    assert(results.contains(op.getContexts()));

    SmallVector<int64_t> selectors;
    if (auto setTy = dyn_cast<SetType>(op.getContexts().getType())) {
      SmallVector<size_t> set;
      for (auto el :
           std::get<SmallVector<LeafEvalValue>>(*results[op.getContexts()]))
        set.push_back(std::get<size_t>(el));
      selectors = {set.begin(), set.end()};
      llvm::sort(selectors);
    } else {
      selectors.push_back(std::get<size_t>(
          std::get<LeafEvalValue>(*results[op.getContexts()])));
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
    results[op->getResult(0)] = std::make_unique<EvalValue>(op.getResourceId());
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

  LogicalResult dispatchOpVisitor(Operation *op) {
    if (op->getDialect() == op->getContext()->getLoadedDialect<RTGDialect>())
      return RTGOpVisitor<Elaborator, LogicalResult>::dispatchOpVisitor(op);

    return ArithOpVisitor<Elaborator, LogicalResult>::dispatchOpVisitor(op);
  }

private:
  std::mt19937 rng;
  DenseMap<Value, std::unique_ptr<EvalValue>> results;
  SymbolTable symTable;
};
} // end namespace

void ElaborationPass::runOnOperation() {
  if (seed.hasValue())
    options.seed = seed;

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
  SymbolTable table(getOperation());
  return Elaborator(table, options).elaborate(testOp);

  // TODO: run DCE because the elaborator does not remove operations that define
  // values.
}

std::unique_ptr<Pass>
rtg::createElaborationPass(const rtg::ElaborationOptions &options) {
  return std::make_unique<ElaborationPass>(options);
}
