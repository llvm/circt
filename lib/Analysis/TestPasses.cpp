//===- TestPasses.cpp - Test passes for the analysis infrastructure -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements test passes for the analysis infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/ControlFlowLoopAnalysis.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt::analysis;

//===----------------------------------------------------------------------===//
// DependenceAnalysis passes.
//===----------------------------------------------------------------------===//

namespace {
struct TestDependenceAnalysisPass
    : public PassWrapper<TestDependenceAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDependenceAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-dependence-analysis"; }
  StringRef getDescription() const override {
    return "Perform dependence analysis and emit results as attributes";
  }
};
} // namespace

void TestDependenceAnalysisPass::runOnOperation() {
  MLIRContext *context = &getContext();

  MemoryDependenceAnalysis analysis(getOperation());

  getOperation().walk([&](Operation *op) {
    if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      return;

    SmallVector<Attribute> deps;

    for (auto dep : analysis.getDependences(op)) {
      if (dep.dependenceType != mlir::DependenceResult::HasDependence)
        continue;

      SmallVector<Attribute> comps;
      for (auto comp : dep.dependenceComponents) {
        SmallVector<Attribute> vector;
        vector.push_back(
            IntegerAttr::get(IntegerType::get(context, 64), *comp.lb));
        vector.push_back(
            IntegerAttr::get(IntegerType::get(context, 64), *comp.ub));
        comps.push_back(ArrayAttr::get(context, vector));
      }

      deps.push_back(ArrayAttr::get(context, comps));
    }

    auto dependences = ArrayAttr::get(context, deps);
    op->setAttr("dependences", dependences);
  });
}

//===----------------------------------------------------------------------===//
// DependenceAnalysis passes.
//===----------------------------------------------------------------------===//

namespace {
struct TestSchedulingAnalysisPass
    : public PassWrapper<TestSchedulingAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSchedulingAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-scheduling-analysis"; }
  StringRef getDescription() const override {
    return "Perform scheduling analysis and emit results as attributes";
  }
};
} // namespace

void TestSchedulingAnalysisPass::runOnOperation() {
  MLIRContext *context = &getContext();

  CyclicSchedulingAnalysis analysis = getAnalysis<CyclicSchedulingAnalysis>();

  getOperation().walk([&](AffineForOp forOp) {
    if (isa<AffineForOp>(forOp.getBody()->front()))
      return;
    CyclicProblem problem = analysis.getProblem(forOp);
    forOp.getBody()->walk([&](Operation *op) {
      for (auto dep : problem.getDependences(op)) {
        assert(!dep.isInvalid());
        if (dep.isAuxiliary())
          op->setAttr("dependence", UnitAttr::get(context));
      }
    });
  });
}

//===----------------------------------------------------------------------===//
// ControlFlowLoopAnalysis passes.
//===----------------------------------------------------------------------===//

namespace {
struct TestControlFlowLoopAnalysisPass
    : public PassWrapper<TestControlFlowLoopAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestControlFlowLoopAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-cf-loop-analysis"; }
  StringRef getDescription() const override {
    return "Perform cf loop analysis and emit results as attributes";
  }
};
} // namespace

static SmallVector<Attribute> &
lookupOrInsert(DenseMap<Block *, SmallVector<Attribute>> &map, Block *key) {
  if (map.count(key) == 0) {
    map.try_emplace(key, SmallVector<Attribute>());
  }
  return map.find(key)->getSecond();
}

void TestControlFlowLoopAnalysisPass::runOnOperation() {
  Region &r = getOperation().getRegion();
  ControlFlowLoopAnalysis analysis(r);
  if (failed(analysis.analyzeRegion())) {
    signalPassFailure();
    return;
  }
  OpBuilder builder(r);
  DenseMap<Block *, SmallVector<Attribute>> blockMap;
  for (const LoopInfo &info : analysis.topLevelLoops) {
    Block *header = info.loopHeader;
    lookupOrInsert(blockMap, header).push_back(builder.getStringAttr("header"));

    for (auto *latch : info.loopLatches)
      lookupOrInsert(blockMap, latch).push_back(builder.getStringAttr("latch"));

    for (auto *inLoop : info.inLoop)
      lookupOrInsert(blockMap, inLoop)
          .push_back(builder.getStringAttr("inLoop"));

    for (auto *exit : info.exitBlocks)
      lookupOrInsert(blockMap, exit).push_back(builder.getStringAttr("exit"));
  }

  for (auto it : blockMap) {
    OperationState opState(builder.getUnknownLoc(), "block.info");
    opState.addAttribute("loopInfo", builder.getArrayAttr(it.getSecond()));
    builder.setInsertionPointToStart(it.getFirst());
    builder.create(opState);
  }
}

//===----------------------------------------------------------------------===//
// InferTopModule passes.
//===----------------------------------------------------------------------===//

namespace {
struct InferTopModulePass
    : public PassWrapper<InferTopModulePass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferTopModulePass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-infer-top-level"; }
  StringRef getDescription() const override {
    return "Perform top level module inference and emit results as attributes "
           "on the enclosing module.";
  }
};
} // namespace

void InferTopModulePass::runOnOperation() {
  circt::hw::InstanceGraph &analysis = getAnalysis<circt::hw::InstanceGraph>();
  auto res = analysis.getInferredTopLevelNodes();
  if (failed(res)) {
    signalPassFailure();
    return;
  }

  llvm::SmallVector<Attribute, 4> attrs;
  for (auto *node : *res)
    attrs.push_back(node->getModule().moduleNameAttr());

  analysis.getParent()->setAttr("test.top",
                                ArrayAttr::get(&getContext(), attrs));
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerAnalysisTestPasses() {
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestDependenceAnalysisPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestSchedulingAnalysisPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestControlFlowLoopAnalysisPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<InferTopModulePass>();
  });
}
} // namespace test
} // namespace circt
