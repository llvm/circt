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
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-to-staticlogic"

using namespace mlir;
using namespace mlir::arith;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::staticlogic;

namespace {

struct AffineToStaticLogic
    : public AffineToStaticLogicBase<AffineToStaticLogic> {
  void runOnFunction() override;

private:
  LogicalResult populateOperatorTypes(SmallVectorImpl<AffineForOp> &loopNest);
  LogicalResult solveSchedulingProblem(SmallVectorImpl<AffineForOp> &loopNest);
  LogicalResult
  createStaticLogicPipeline(SmallVectorImpl<AffineForOp> &loopNest);

  CyclicSchedulingAnalysis *schedulingAnalysis;
};

} // namespace

void AffineToStaticLogic::runOnFunction() {
  // Get scheduling analysis for the whole function.
  schedulingAnalysis = &getAnalysis<CyclicSchedulingAnalysis>();

  // Collect perfectly nested loops and work on them.
  auto outerLoops = getOperation().getOps<AffineForOp>();
  for (auto root : llvm::make_early_inc_range(outerLoops)) {
    SmallVector<AffineForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, root);

    // Restrict to single loops to simplify things for now.
    if (nestedLoops.size() != 1)
      continue;

    // Populate the target operator types.
    if (failed(populateOperatorTypes(nestedLoops)))
      return signalPassFailure();

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveSchedulingProblem(nestedLoops)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createStaticLogicPipeline(nestedLoops)))
      return signalPassFailure();
  }
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult AffineToStaticLogic::populateOperatorTypes(
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
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AddIOp, AffineIfOp, AffineYieldOp, mlir::ConstantOp, IndexCastOp,
              memref::AllocaOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<AffineReadOpInterface, AffineWriteOpInterface>(
            [&](Operation *seqOp) {
              // Some known sequential ops. In certain cases, reads may be
              // combinational in Calyx, but taking advantage of that is left as
              // a future enhancement.
              problem.setLinkedOperatorType(seqOp, seqOpr);
              return WalkResult::advance();
            })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Some known multi-cycle ops.
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return forOp.emitError("unsupported operation ") << *unsupported;

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult AffineToStaticLogic::solveSchedulingProblem(
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
    return failure();

  auto *anchor = forOp.getBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

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

  return success();
}

/// Create the pipeline op for a loop nest.
LogicalResult AffineToStaticLogic::createStaticLogicPipeline(
    SmallVectorImpl<AffineForOp> &loopNest) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loopNest.back();

  // Retrieve the cyclic scheduling problem for this loop.
  CyclicProblem &problem = schedulingAnalysis->getProblem(forOp);

  auto outerLoop = loopNest.front();
  auto innerLoop = loopNest.back();
  ImplicitLocOpBuilder builder(outerLoop.getLoc(), outerLoop);

  // Create constants for the loop's lower and upper bounds.
  int64_t lbValue = innerLoop.getConstantLowerBound();
  auto lowerBound = builder.create<arith::ConstantOp>(
      IntegerAttr::get(builder.getI64Type(), lbValue));
  int64_t ubValue = innerLoop.getConstantUpperBound();
  auto upperBound = builder.create<arith::ConstantOp>(
      IntegerAttr::get(builder.getI64Type(), ubValue));
  int64_t stepValue = innerLoop.getStep();
  auto step = builder.create<arith::ConstantOp>(
      IntegerAttr::get(builder.getI64Type(), stepValue));

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = innerLoop.getResultTypes();

  auto ii =
      builder.getI64IntegerAttr(problem.getInitiationInterval().getValue());

  SmallVector<Value> iterArgs;
  iterArgs.push_back(lowerBound);
  iterArgs.append(innerLoop.getIterOperands().begin(),
                  innerLoop.getIterOperands().end());

  auto pipeline = builder.create<PipelineWhileOp>(resultTypes, ii, iterArgs);

  // Create the condition, which currently just compares the induction variable
  // to the upper bound.
  Block &condBlock = pipeline.getCondBlock();
  builder.setInsertionPointToStart(&condBlock);
  auto cmpResult = builder.create<arith::CmpIOp>(
      builder.getI1Type(), arith::CmpIPredicate::ult, condBlock.getArgument(0),
      upperBound);
  condBlock.getTerminator()->insertOperands(0, {cmpResult});

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    auto startTime = problem.getStartTime(op);
    if (!startTime.hasValue())
      return op->emitOpError("didn't have scheduled start time");
    if (isa<AffineYieldOp>(op))
      continue;
    startGroups[*startTime].push_back(op);
  }

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  BlockAndValueMapping valueMap;
  BlockAndValueMapping resultMap;
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(forOp.getBody()->getArgument(i),
                 pipeline.getStagesBlock().getArgument(i));

  // Create the stages.
  Block &stagesBlock = pipeline.getStagesBlock();
  builder.setInsertionPointToStart(&stagesBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (auto group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    OpBuilder::InsertionGuard g(builder);

    // Collect the return types for this stage.
    SmallVector<Type> stageTypes;
    for (auto *op : group)
      stageTypes.append(op->getResultTypes().begin(),
                        op->getResultTypes().end());

    // Add the induction variable increment in the first stage.
    if (startTime == 0)
      stageTypes.push_back(lowerBound.getType());

    // Create the stage itself.
    auto stage = builder.create<PipelineStageOp>(stageTypes);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    // Sort the group according to original dominance.
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });

    // Move over the operations.
    for (auto *op : group) {
      unsigned resultIndex = stageTerminator->getNumOperands();
      auto *newOp = builder.clone(*op, valueMap);

      stageTerminator->insertOperands(resultIndex, newOp->getResults());

      // Add the stage results to the value map for the original op.
      for (size_t i = 0; i < newOp->getNumResults(); ++i) {
        auto newValue = stage->getResult(resultIndex + i);
        auto oldValue = op->getResult(i);
        resultMap.map(oldValue, newValue);
      }
    }

    // Add the induction variable increment to the first stage.
    if (startTime == 0) {
      auto incResult =
          builder.create<arith::AddIOp>(stagesBlock.getArgument(0), step);
      stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                      incResult->getResults());
    }
  }

  // Add the iter args and results to the terminator.
  auto stagesTerminator =
      cast<PipelineTerminatorOp>(stagesBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  termIterArgs.push_back(
      stagesBlock.front().getResult(stagesBlock.front().getNumResults() - 1));
  for (auto value : forOp.getBody()->getTerminator()->getOperands()) {
    termIterArgs.push_back(resultMap.lookup(value));
    termResults.push_back(resultMap.lookup(value));
  }

  stagesTerminator.iter_argsMutable().append(termIterArgs);
  stagesTerminator.resultsMutable().append(termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  // Remove the loop nest from the IR.
  loopNest.front().walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

std::unique_ptr<mlir::Pass> circt::createAffineToStaticLogic() {
  return std::make_unique<AffineToStaticLogic>();
}
