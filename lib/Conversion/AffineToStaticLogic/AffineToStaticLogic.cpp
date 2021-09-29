//===- AffineToStaticlogic.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToStaticLogic/AffineToStaticLogic.h"
#include "../PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-to-staticlogic"

using namespace mlir;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace {

struct AffineToStaticLogic
    : public AffineToStaticLogicBase<AffineToStaticLogic> {
  void runOnFunction() override;

private:
  void runOnAffineFor(AffineForOp forOp,
                      MemoryDependenceAnalysis memoryAnalysis);
};

} // namespace

void AffineToStaticLogic::runOnFunction() {
  MemoryDependenceAnalysis memoryAnalysis =
      getAnalysis<MemoryDependenceAnalysis>();
  getFunction().walk(
      [&](AffineForOp forOp) { runOnAffineFor(forOp, memoryAnalysis); });
}

void AffineToStaticLogic::runOnAffineFor(
    AffineForOp forOp, MemoryDependenceAnalysis memoryAnalysis) {
  // Only consider innermost AffineForOps.
  if (isa<AffineForOp>(forOp.getBody()->front()))
    return;

  // Create a cyclic scheduling problem.
  CyclicProblem problem(forOp);

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of combinational and memory operators for now.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);

  Operation *unsupported;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    problem.insertOperation(op);

    // Some known combinational ops.
    if (isa<AddIOp, AffineIfOp, AffineYieldOp, ConstantOp, IndexCastOp,
            memref::AllocaOp>(op)) {
      problem.setLinkedOperatorType(op, combOpr);
      return WalkResult::advance();
    }

    // Some known sequential ops.
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
      problem.setLinkedOperatorType(op, seqOpr);
      return WalkResult::advance();
    }

    unsupported = op;
    return WalkResult::interrupt();
  });

  if (result.wasInterrupted()) {
    forOp.emitError("unsupported operation ") << *unsupported;
    return signalPassFailure();
  }

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    ArrayRef<MemoryDependence> dependences = memoryAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (MemoryDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      assert(succeeded(problem.insertDependence(dep)));

      // Find the greatest distance lower bound from any loop and use that for
      // this dependence.
      unsigned distance = 0;
      for (DependenceComponent comp : memoryDep.dependenceComponents)
        if (comp.lb.getValue() > distance)
          distance = comp.lb.getValue();

      problem.setDistance(dep, distance);
    }
  });

  // Insert conditional dependences into the problem.
  forOp.getBody()->walk([&](AffineIfOp op) {
    // Insert a dependence from the if to all ops in the then region.
    for (auto &inner : *op.getThenBlock()) {
      Problem::Dependence dep(op, &inner);
      assert(succeeded(problem.insertDependence(dep)));
    }

    // Insert a dependence from the if to all ops in the else region.
    if (op.hasElse()) {
      for (auto &inner : *op.getElseBlock()) {
        Problem::Dependence dep(op, &inner);
        assert(succeeded(problem.insertDependence(dep)));
      }
    }
  });

  // Verify and solve the scheduling problem, and optionally debug it.
#ifndef NDEBUG
  if (llvm::isCurrentDebugType(DEBUG_TYPE))
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      llvm::dbgs() << "Scheduling inputs for " << *op;
      auto opr = problem.getLinkedOperatorType(op);
      llvm::dbgs() << "\nopr = " << opr;
      llvm::dbgs() << "\nlatency = " << problem.getLatency(*opr);
      for (auto dep : problem.getDependences(op))
        if (auto distance = problem.getDistance(dep); distance.hasValue())
          llvm::dbgs() << "\ndep = { distance = " << distance
                       << ", source = " << *dep.getSource() << '}';
      llvm::dbgs() << "\n\n";
    });
#endif

  if (failed(problem.check()))
    return signalPassFailure();

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](AffineWriteOpInterface op) {
    Problem::Dependence dep(op, anchor);
    assert(succeeded(problem.insertDependence(dep)));
  });

  if (failed(scheduleSimplex(problem, anchor)))
    return signalPassFailure();

#ifndef NDEBUG
  if (llvm::isCurrentDebugType(DEBUG_TYPE)) {
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\nstart = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  }
#endif
}

std::unique_ptr<mlir::Pass> circt::createAffineToStaticLogic() {
  return std::make_unique<AffineToStaticLogic>();
}
