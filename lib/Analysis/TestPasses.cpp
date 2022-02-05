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

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt::analysis;

//===----------------------------------------------------------------------===//
// DependenceAnalysis passes.
//===----------------------------------------------------------------------===//

namespace {
struct TestDependenceAnalysisPass
    : public PassWrapper<TestDependenceAnalysisPass, OperationPass<FuncOp>> {
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
        vector.push_back(IntegerAttr::get(IntegerType::get(context, 64),
                                          comp.lb.getValue()));
        vector.push_back(IntegerAttr::get(IntegerType::get(context, 64),
                                          comp.ub.getValue()));
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
    : public PassWrapper<TestSchedulingAnalysisPass, OperationPass<FuncOp>> {
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
}
} // namespace test
} // namespace circt
