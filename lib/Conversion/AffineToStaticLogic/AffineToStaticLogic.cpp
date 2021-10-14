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
                      CyclicSchedulingAnalysis schedulingAnalysis);
};

} // namespace

void AffineToStaticLogic::runOnFunction() {
  CyclicSchedulingAnalysis schedulingAnalysis =
      getAnalysis<CyclicSchedulingAnalysis>();
  getFunction().walk(
      [&](AffineForOp forOp) { runOnAffineFor(forOp, schedulingAnalysis); });
}

void AffineToStaticLogic::runOnAffineFor(
    AffineForOp forOp, CyclicSchedulingAnalysis schedulingAnalysis) {
  // Only consider innermost AffineForOps.
  if (isa<AffineForOp>(forOp.getBody()->front()))
    return;

  // Create a cyclic scheduling problem.
  CyclicProblem problem = schedulingAnalysis.getProblem(forOp);

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of combinational and memory operators for now.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);

  Operation *unsupported;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
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

  // Verify and solve the scheduling problem, and optionally debug it.
#ifndef NDEBUG
  if (llvm::isCurrentDebugType(DEBUG_TYPE))
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      llvm::dbgs() << "Scheduling inputs for " << *op;
      auto opr = problem.getLinkedOperatorType(op);
      llvm::dbgs() << "\nopr = " << opr;
      llvm::dbgs() << "\nlatency = " << problem.getLatency(*opr);
      for (auto dep : problem.getDependences(op))
        if (dep.isAuxiliary())
          llvm::dbgs() << "\ndep = { distance = " << problem.getDistance(dep)
                       << ", source = " << *dep.getSource() << '}';
      llvm::dbgs() << "\n\n";
    });
#endif

  if (failed(problem.check()))
    return signalPassFailure();

  auto *anchor = forOp.getBody()->getTerminator();
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
