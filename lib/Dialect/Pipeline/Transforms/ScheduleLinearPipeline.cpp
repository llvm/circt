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
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/Support/Debug.h"

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
  return isa<hw::ConstantOp, pipeline::ReturnOp>(op);
}

void ScheduleLinearPipelinePass::runOnOperation() {
  auto pipeline = getOperation();

  auto stageOpIt = pipeline.getOps<PipelineStageOp>();
  auto stageRegOpIt = pipeline.getOps<PipelineStageRegisterOp>();

  if ((std::distance(stageOpIt.begin(), stageOpIt.end()) != 0) ||
      (std::distance(stageRegOpIt.begin(), stageRegOpIt.end()) != 0)) {
    pipeline.emitError("Pipeline cannot have any stages or stage registers.");
    return signalPassFailure();
  }

  // Load operator info from attribute.
  auto problem = SharedOperatorsProblem::get(pipeline);
  auto operatorInfo = getOperatorInfo(pipeline);
  if (!operatorInfo) {
    pipeline.emitError() << "Expected 'scheduling.operator_info' attribute on "
                            "pipeline.";
    return signalPassFailure();
  }
  DenseMap<StringAttr, Problem::OperatorType> operatorTypes;
  for (auto opInfo : *operatorInfo) {
    auto opName = opInfo.first;
    auto opType = problem.getOrInsertOperatorType(opName.strref());
    problem.setLatency(opType, opInfo.second.latency.getValue().getZExtValue());
    operatorTypes[opName] = opType;
  }

  // Set operation operator types.
  Operation *lastOp = nullptr;
  for (auto &op : pipeline.getOps()) {
    // Skip if is a known non-functional operator
    if (ignoreOp(&op))
      continue;

    // Lookup operator info.
    auto opName = op.getName().getIdentifier();
    auto info = operatorTypes.find(opName);
    if (info == operatorTypes.end()) {
      op.emitError() << "Operator info for operation '" << opName
                     << " not found in 'scheduling.operator_info' "
                        "attribute.";
      return signalPassFailure();
    }
    problem.insertOperation(&op);
    problem.setLinkedOperatorType(&op, info->second);
    lastOp = &op;
  }

  // Solve!
  assert(succeeded(problem.check()));
  if (failed(scheduling::scheduleSimplex(problem, lastOp))) {
    pipeline.emitError("Failed to schedule pipeline.");
    return signalPassFailure();
  }

  // Gather stage results.
  using StageIdx = unsigned;
  // Maintain stage map as an ordered map.
  std::map<StageIdx, llvm::SmallVector<Operation *>> stageMap;
  DenseMap<StageIdx, PipelineStageOp> stages;
  llvm::SmallVector<Operation *, 4> otherOps;
  for (auto &op : pipeline.getOps()) {
    if (ignoreOp(&op)) {
      if (!isa<pipeline::ReturnOp>(&op))
        otherOps.push_back(&op);
      continue;
    }
    unsigned startTime = *problem.getStartTime(&op);
    stageMap[startTime].push_back(&op);
  }

  // Build pipeline stages.
  OpBuilder b(pipeline.getBody());
  auto loc = pipeline.getLoc();
  Value stageValid = b.create<hw::ConstantOp>(loc, APInt(1, 1));
  StageIdx numStages = stageMap.rbegin()->first;
  for (unsigned i = 0; i <= numStages; ++i) {
    auto nextStage = b.create<PipelineStageOp>(loc, stageValid);
    stageValid = nextStage.getValid();
    stages[i] = nextStage;
  }

  // Reorder pipeline. Initially place unscheduled ops at the start, and then
  // all following ops in their assigned stage.
  for (auto op : otherOps)
    op->moveBefore(stages[0]);

  for (auto [stage, ops] : stageMap) {
    auto stageOp = stages.find(stage);
    for (auto op : ops)
      op->moveBefore(stageOp->second);
  }
}

std::unique_ptr<mlir::Pass>
circt::pipeline::createScheduleLinearPipelinePass() {
  return std::make_unique<ScheduleLinearPipelinePass>();
}
