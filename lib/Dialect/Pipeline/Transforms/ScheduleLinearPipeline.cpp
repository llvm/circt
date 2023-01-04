//===- ScheduleLinearPipeline.cpp - Linear pipeline scheduling pass -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions linear pipeline scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"
#include "circt/Support/BackedgeBuilder.h"

#define DEBUG_TYPE "pipeline-schedule-linear"

using namespace mlir;
using namespace circt;
using namespace pipeline;

namespace {

class ScheduleLinearPipelinePass
    : public ScheduleLinearPipelineBase<ScheduleLinearPipelinePass> {
public:
  void runOnOperation() override;
};

} // end anonymous namespace

// Returns true if 'op' should be ignored in the scheduling problem.
static bool ignoreOp(Operation *op) {
  return op->hasTrait<OpTrait::ConstantLike>();
}

void ScheduleLinearPipelinePass::runOnOperation() {
  auto pipeline = getOperation();

  // Get operator library for the pipeline.
  auto opLibAttr = pipeline->getAttrOfType<FlatSymbolRefAttr>("operator_lib");
  if (!opLibAttr) {
    pipeline.emitError("missing 'operator_lib' attribute");
    return signalPassFailure();
  }
  auto opLib = dyn_cast_or_null<ssp::OperatorLibraryOp>(
      SymbolTable::lookupNearestSymbolFrom(pipeline->getParentOp(), opLibAttr));
  if (!opLib) {
    pipeline.emitError("operator library '") << opLibAttr << "' not found";
    return signalPassFailure();
  }

  auto stageOpIt = pipeline.getOps<PipelineStageOp>();
  auto stageRegOpIt = pipeline.getOps<PipelineStageRegisterOp>();

  if (stageOpIt.begin() != stageOpIt.end() ||
      stageRegOpIt.begin() != stageRegOpIt.end()) {
    pipeline.emitError("Pipeline cannot have any stages or stage registers.");
    return signalPassFailure();
  }

  // Load operator info from attribute.
  auto problem = Problem::get(pipeline);

  DenseMap<SymbolRefAttr, Problem::OperatorType> operatorTypes;
  SmallDenseMap<StringAttr, unsigned> oprIds;

  // Set operation operator types.
  auto returnOp =
      cast<pipeline::ReturnOp>(pipeline.getBodyBlock()->getTerminator());
  for (auto &op : pipeline.getOps()) {
    // Skip if is a known non-functional operator
    if (ignoreOp(&op))
      continue;

    Problem::OperatorType operatorType;
    bool isReturnOp = &op == returnOp.getOperation();
    if (isReturnOp) {
      // Construct an operator type for the return op (not an externally defined
      // operator type since it is intrinsic to this pass).
      operatorType = problem.getOrInsertOperatorType("return");
      problem.setLatency(operatorType, 0);
    } else {
      // Lookup operator info.
      auto operatorTypeAttr =
          op.getAttrOfType<SymbolRefAttr>("ssp.operator_type");
      if (!operatorTypeAttr) {
        op.emitError()
            << "Expected 'ssp.operator_type' attribute on operation.";
        return signalPassFailure();
      }

      auto operatorTypeIt = operatorTypes.find(operatorTypeAttr);
      if (operatorTypeIt == operatorTypes.end()) {
        // First time seeing operator type - load it into the problem.
        auto opTypeOp =
            opLib.lookupSymbol<ssp::OperatorTypeOp>(operatorTypeAttr);
        if (!opTypeOp) {
          op.emitError() << "Operator type '" << operatorTypeAttr
                         << "' not found in operator library.";
          return signalPassFailure();
        }

        auto insertRes = operatorTypes.try_emplace(
            operatorTypeAttr, ssp::loadOperatorType<Problem, ssp::LatencyAttr>(
                                  problem, opTypeOp, oprIds));
        operatorTypeIt = insertRes.first;
      }
      operatorType = operatorTypeIt->second;
    }

    problem.insertOperation(&op);
    problem.setLinkedOperatorType(&op, operatorType);

    // We want the return op to be a sink node for the dependence graph, i.e. it
    // should (transitively) depend on every other op. This is done by inserting
    // auxiliary dependences from ops without users, complementing the implicit
    // dependences from the return op's operands.
    if (!isReturnOp && op.use_empty()) {
      if (failed(problem.insertDependence({&op, returnOp.getOperation()}))) {
        op.emitError()
            << "Failed to insert dependence from operation to return op.";
        return signalPassFailure();
      }
    }
  }

  // Solve!
  assert(succeeded(problem.check()));
  if (failed(scheduling::scheduleSimplex(problem, returnOp.getOperation()))) {
    pipeline.emitError("Failed to schedule pipeline.");
    return signalPassFailure();
  }
  assert(succeeded(problem.verify()));

  // Gather stage results.
  using StageIdx = unsigned;

  // Maintain a mapping of {start time : [operations]}, that contains the
  // operations scheduled to a given start time. This is an ordered map, so that
  // we can iterate over the stages in order.
  std::map<StageIdx, llvm::SmallVector<Operation *>> stageMap;
  DenseMap<StageIdx, PipelineStageOp> stages;
  llvm::SmallVector<Operation *, 4> otherOps;

  // Iterate over the ops in the pipeline, and add them to the stage map.
  // While doing so, we also build the pipeline stage operations.
  OpBuilder b(pipeline.getBody());
  auto loc = pipeline.getLoc();
  unsigned currentEndTime = 0;
  Value stageValid = b.create<hw::ConstantOp>(loc, APInt(1, 1));
  for (auto &op : pipeline.getOps()) {
    if (ignoreOp(&op)) {
      otherOps.push_back(&op);
      continue;
    }
    unsigned startTime = *problem.getStartTime(&op);
    stageMap[startTime].push_back(&op);

    auto oldEndTime = currentEndTime;
    currentEndTime = std::max(currentEndTime, *problem.getEndTime(&op));
    for (unsigned i = oldEndTime; i < currentEndTime; ++i) {
      auto nextStage = b.create<PipelineStageOp>(loc, stageValid);
      stageValid = nextStage.getValid();
      stages[i] = nextStage;
    }
  }

  // Reorder pipeline. Initially place unscheduled ops at the start, and then
  // all following ops in their assigned stage.
  for (auto *op : otherOps)
    op->moveBefore(stages[0]);

  for (auto [startTime, ops] : stageMap) {
    auto stageOp = stages.find(startTime);
    Operation *moveBeforeOp;
    if (stageOp == stages.end())
      // Operation in "last" stage (move after last stage)
      moveBeforeOp = returnOp;
    else
      // Operation before a specific stage (move before stage)
      moveBeforeOp = stageOp->second;

    for (auto *op : ops)
      op->moveBefore(moveBeforeOp);
  }

  // Replace the pipeline return value with one that uses the last stage valid
  // signal.
  b.setInsertionPoint(returnOp);
  b.create<pipeline::ReturnOp>(returnOp.getLoc(), returnOp.getOutputs(),
                               stageValid);
  returnOp.erase();
}

std::unique_ptr<mlir::Pass>
circt::pipeline::createScheduleLinearPipelinePass() {
  return std::make_unique<ScheduleLinearPipelinePass>();
}
