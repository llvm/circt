//===- AffineToStaticlogic.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToStaticLogic.h"
#include "../PassDetail.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-to-staticlogic"

using namespace mlir;
using namespace mlir::arith;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace {

struct AffineToStaticLogic
    : public AffineToStaticLogicBase<AffineToStaticLogic> {
  void runOnFunction() override;

private:
  void populateOperatorTypes(SmallVectorImpl<AffineForOp> &loopNest);
  void solveSchedulingProblem(SmallVectorImpl<AffineForOp> &loopNest);

  CyclicSchedulingAnalysis *schedulingAnalysis;
};

} // namespace

void AffineToStaticLogic::runOnFunction() {
  // Get scheduling analysis for the whole function.
  schedulingAnalysis = &getAnalysis<CyclicSchedulingAnalysis>();

  // Collect perfectly nested loops and work on them.
  for (auto root : getOperation().getOps<AffineForOp>()) {
    SmallVector<AffineForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, root);
    populateOperatorTypes(nestedLoops);
    solveSchedulingProblem(nestedLoops);
  }
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
void AffineToStaticLogic::populateOperatorTypes(
    SmallVectorImpl<AffineForOp> &loopNest) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loopNest.back();

  // Retrieve the cyclic scheduling problem for this loop.
  CyclicProblem &problem = schedulingAnalysis->getProblem(forOp);

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);

  Operation *unsupported;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    // Some known combinational ops.
    if (isa<AddIOp, AffineIfOp, AffineYieldOp, mlir::ConstantOp, IndexCastOp,
            memref::AllocaOp>(op)) {
      problem.setLinkedOperatorType(op, combOpr);
      return WalkResult::advance();
    }

    // Some known sequential ops.
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
      problem.setLinkedOperatorType(op, seqOpr);
      return WalkResult::advance();
    }

    // Some known multi-cycle ops.
    if (isa<MulIOp>(op)) {
      problem.setLinkedOperatorType(op, mcOpr);
      return WalkResult::advance();
    }

    unsupported = op;
    return WalkResult::interrupt();
  });

  if (result.wasInterrupted()) {
    forOp.emitError("unsupported operation ") << *unsupported;
    return signalPassFailure();
  }
}

/// Solve the pre-computed scheduling problem.
void AffineToStaticLogic::solveSchedulingProblem(
    SmallVectorImpl<AffineForOp> &loopNest) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loopNest.back();

  // Retrieve the cyclic scheduling problem for this loop.
  CyclicProblem &problem = schedulingAnalysis->getProblem(forOp);

  // Optionally debug problem inputs.
  LLVM_DEBUG(forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::dbgs() << "Scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr;
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return signalPassFailure();

  auto *anchor = forOp.getBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return signalPassFailure();

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });
}

std::unique_ptr<mlir::Pass> circt::createAffineToStaticLogic() {
  return std::make_unique<AffineToStaticLogic>();
}
