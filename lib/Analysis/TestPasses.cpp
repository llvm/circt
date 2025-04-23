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

#include "circt/Analysis/DebugAnalysis.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Analysis/OpCountAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::dataflow;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;

//===----------------------------------------------------------------------===//
// DebugAnalysis
//===----------------------------------------------------------------------===//

namespace {
struct TestDebugAnalysisPass
    : public PassWrapper<TestDebugAnalysisPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDebugAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-debug-analysis"; }
  StringRef getDescription() const override {
    return "Perform debug analysis and emit results as attributes";
  }
};
} // namespace

void TestDebugAnalysisPass::runOnOperation() {
  auto *context = &getContext();
  auto &analysis = getAnalysis<DebugAnalysis>();
  for (auto *op : analysis.debugOps) {
    op->setAttr("debug.only", UnitAttr::get(context));
  }
}

//===----------------------------------------------------------------------===//
// DependenceAnalysis
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
      if (dep.dependenceType != DependenceResult::HasDependence)
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
// SchedulingAnalysis
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
// InstanceGraph
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
    attrs.push_back(node->getModule().getModuleNameAttr());

  analysis.getParent()->setAttr("test.top",
                                ArrayAttr::get(&getContext(), attrs));
}

//===----------------------------------------------------------------------===//
// FIRRTL Instance Info
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLInstanceInfoPass
    : public PassWrapper<FIRRTLInstanceInfoPass,
                         OperationPass<firrtl::CircuitOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRRTLInstanceInfoPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-firrtl-instance-info"; }
  StringRef getDescription() const override {
    return "Run firrtl::InstanceInfo analysis and show the results.  This pass "
           "is intended to be used for testing purposes only.";
  }
};
} // namespace

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const bool a) {
  if (a)
    return os << "true";
  return os << "false";
}

static void printCircuitInfo(firrtl::CircuitOp op,
                             firrtl::InstanceInfo &iInfo) {
  OpPrintingFlags flags;
  flags.skipRegions();
  llvm::errs() << "  - operation: ";
  op->print(llvm::errs(), flags);
  llvm::errs() << "\n"
               << "    hasDut: " << iInfo.hasDut() << "\n"
               << "    dut: ";
  if (auto dutNode = iInfo.getDut())
    dutNode->print(llvm::errs(), flags);
  else
    llvm::errs() << "null";
  llvm::errs() << "\n"
               << "    effectiveDut: ";
  iInfo.getEffectiveDut()->print(llvm::errs(), flags);
  llvm::errs() << "\n";
}

static void printModuleInfo(igraph::ModuleOpInterface op,
                            firrtl::InstanceInfo &iInfo) {
  OpPrintingFlags flags;
  flags.skipRegions();
  llvm::errs() << "  - operation: ";
  op->print(llvm::errs(), flags);
  llvm::errs() << "\n"
               << "    isDut: " << iInfo.isDut(op) << "\n"
               << "    anyInstanceUnderDut: " << iInfo.anyInstanceUnderDut(op)
               << "\n"
               << "    allInstancesUnderDut: " << iInfo.allInstancesUnderDut(op)
               << "\n"
               << "    anyInstanceUnderEffectiveDut: "
               << iInfo.anyInstanceUnderEffectiveDut(op) << "\n"
               << "    allInstancesUnderEffectiveDut: "
               << iInfo.allInstancesUnderEffectiveDut(op) << "\n"
               << "    anyInstanceUnderLayer: "
               << iInfo.anyInstanceUnderLayer(op) << "\n"
               << "    allInstancesUnderLayer: "
               << iInfo.allInstancesUnderLayer(op) << "\n"
               << "    anyInstanceInDesign: " << iInfo.anyInstanceInDesign(op)
               << "\n"
               << "    allInstancesInDesign: " << iInfo.allInstancesInDesign(op)
               << "\n"
               << "    anyInstanceInEffectiveDesign: "
               << iInfo.anyInstanceInEffectiveDesign(op) << "\n"
               << "    allInstancesInEffectiveDesign: "
               << iInfo.allInstancesInEffectiveDesign(op) << "\n";
}

void FIRRTLInstanceInfoPass::runOnOperation() {
  auto &iInfo = getAnalysis<firrtl::InstanceInfo>();

  printCircuitInfo(getOperation(), iInfo);
  for (auto op :
       getOperation().getBodyBlock()->getOps<igraph::ModuleOpInterface>())
    printModuleInfo(op, iInfo);
}

//===----------------------------------------------------------------------===//
// Comb IntRange Analysis
//===----------------------------------------------------------------------===//

namespace {
struct TestCombIntegerRangeAnalysisPass
    : public PassWrapper<TestCombIntegerRangeAnalysisPass,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCombIntegerRangeAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override {
    return "test-comb-intrange-analysis";
  }
  StringRef getDescription() const override {
    return "Perform integer range analysis on comb dialect and set results as "
           "attributes";
  }
};
} // namespace

void TestCombIntegerRangeAnalysisPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
  op->walk([&](Operation *op) {
    for (auto value : op->getResults()) {
      if (auto *range = solver.lookupState<IntegerValueRangeLattice>(value)) {
        assert(op->getResults().size() == 1 &&
               "Expected a single result for the operation analysis");
        assert(!range->getValue().isUninitialized() &&
               "Expected a valid range for the value");
        auto interval = range->getValue().getValue();
        auto umin = interval.umin();
        auto uminAttr =
            IntegerAttr::get(IntegerType::get(ctx, umin.getBitWidth()), umin);
        auto umax = interval.umax();
        auto umaxAttr =
            IntegerAttr::get(IntegerType::get(ctx, umax.getBitWidth()), umax);
        auto smin = interval.umin();
        auto sminAttr =
            IntegerAttr::get(IntegerType::get(ctx, smin.getBitWidth()), smin);
        auto smax = interval.umax();
        auto smaxAttr =
            IntegerAttr::get(IntegerType::get(ctx, smax.getBitWidth()), smax);
        op->setAttr("integer.urange",
                    ArrayAttr::get(ctx, {uminAttr, umaxAttr}));
        op->setAttr("integer.srange",
                    ArrayAttr::get(ctx, {sminAttr, smaxAttr}));
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerAnalysisTestPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestDependenceAnalysisPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestSchedulingAnalysisPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestDebugAnalysisPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<InferTopModulePass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<FIRRTLInstanceInfoPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestCombIntegerRangeAnalysisPass>();
  });
}
} // namespace test
} // namespace circt
