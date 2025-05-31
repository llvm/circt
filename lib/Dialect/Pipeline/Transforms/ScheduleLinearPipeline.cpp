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

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/Pipeline/PipelineOps.h"
#include "circt/Dialect/Pipeline/PipelinePasses.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "pipeline-schedule-linear"

namespace circt {
namespace pipeline {
#define GEN_PASS_DEF_SCHEDULELINEARPIPELINE
#include "circt/Dialect/Pipeline/PipelinePasses.h.inc"
} // namespace pipeline
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::scheduling;
using namespace pipeline;

namespace {

class ScheduleLinearPipelinePass
    : public circt::pipeline::impl::ScheduleLinearPipelineBase<
          ScheduleLinearPipelinePass> {
public:
  void runOnOperation() override;

private:
  LogicalResult schedulePipeline(UnscheduledPipelineOp pipeline);
};

} // end anonymous namespace

// Returns true if 'op' should be ignored in the scheduling problem.
static bool ignoreOp(Operation *op) {
  return op->hasTrait<OpTrait::ConstantLike>();
}

LogicalResult
ScheduleLinearPipelinePass::schedulePipeline(UnscheduledPipelineOp pipeline) {
  // Get operator library for the pipeline - assume it's placed in the top level
  // module.
  auto opLibAttr = pipeline->getAttrOfType<FlatSymbolRefAttr>("operator_lib");
  if (!opLibAttr)
    return pipeline.emitError("missing 'operator_lib' attribute");
  auto parentModule = pipeline->getParentOfType<ModuleOp>();
  auto opLib = parentModule.lookupSymbol<ssp::OperatorLibraryOp>(opLibAttr);
  if (!opLib)
    return pipeline.emitError("operator library '")
           << opLibAttr << "' not found";

  // Load operator info from attribute.
  Problem problem(pipeline);

  DenseMap<SymbolRefAttr, Problem::OperatorType> operatorTypes;
  SmallDenseMap<ssp::OperatorType, unsigned> oprIds;

  // Set operation operator types.
  auto returnOp =
      cast<pipeline::ReturnOp>(pipeline.getEntryStage()->getTerminator());
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
      if (!operatorTypeAttr)
        return op.emitError()
               << "Expected 'ssp.operator_type' attribute on operation.";

      auto operatorTypeIt = operatorTypes.find(operatorTypeAttr);
      if (operatorTypeIt == operatorTypes.end()) {
        // First time seeing operator type - load it into the problem.
        auto opTypeOp =
            opLib.lookupSymbol<ssp::OperatorTypeOp>(operatorTypeAttr);
        if (!opTypeOp)
          return op.emitError() << "Operator type '" << operatorTypeAttr
                                << "' not found in operator library.";

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
      if (failed(problem.insertDependence({&op, returnOp.getOperation()})))
        return op.emitError()
               << "Failed to insert dependence from operation to return op.";
    }
  }

  // Solve!
  assert(succeeded(problem.check()));
  if (failed(scheduling::scheduleSimplex(problem, returnOp.getOperation())))
    return pipeline.emitError("Failed to schedule pipeline.");

  assert(succeeded(problem.verify()));

  // Gather stage results.
  using StageIdx = unsigned;

  OpBuilder b(pipeline.getContext());

  // Maintain a mapping of {start time : [operations]}, that contains the
  // operations scheduled to a given start time. This is an ordered map, so that
  // we can iterate over the stages in order.
  std::map<StageIdx, llvm::SmallVector<Operation *>> stageMap;
  llvm::SmallVector<Operation *, 4> otherOps;

  // Create the scheduled pipeline.
  b.setInsertionPoint(pipeline);
  auto schedPipeline = b.template create<pipeline::ScheduledPipelineOp>(
      pipeline.getLoc(), pipeline.getDataOutputs().getTypes(),
      pipeline.getInputs(), pipeline.getInputNames(), pipeline.getOutputNames(),
      pipeline.getClock(), pipeline.getGo(), pipeline.getReset(),
      pipeline.getStall(), pipeline.getNameAttr());

  Block *currentStage = schedPipeline.getStage(0);

  for (auto [oldBArg, newBArg] :
       llvm::zip(pipeline.getEntryStage()->getArguments(),
                 currentStage->getArguments()))
    oldBArg.replaceAllUsesWith(newBArg);

  // Iterate over the ops in the pipeline, and add them to the stage map.
  // While doing so, we also build the pipeline stage operations.
  unsigned currentEndTime = 0;
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
      Block *newStage = schedPipeline.addStage();

      // Create a StageOp in the new stage, and branch it to the newly created
      // stage.
      b.setInsertionPointToEnd(currentStage);
      b.create<pipeline::StageOp>(pipeline.getLoc(), newStage, ValueRange{},
                                  ValueRange{});
      currentStage = newStage;
    }
  }

  // Move the return op to the last stage in the scheduled pipeline.
  returnOp->moveBefore(currentStage, currentStage->end());

  // Reorder pipeline. Initially place unscheduled ops at the entry stage, and
  // then all following ops in their assigned stage.
  Block *entryStage = schedPipeline.getStage(0);
  Operation *entryStageTerminator = entryStage->getTerminator();
  for (auto *op : otherOps)
    op->moveBefore(entryStageTerminator);

  for (auto [startTime, ops] : stageMap) {
    Block *stage = schedPipeline.getStage(startTime);

    // Caching of SourceOp passthrough values defined in this stage.
    mlir::DenseMap<Value, Value> sourceOps;
    auto getOrCreateSourceOp = [&](OpOperand &opOperand) {
      Value v = opOperand.get();
      auto it = sourceOps.find(v);
      if (it == sourceOps.end()) {
        b.setInsertionPoint(opOperand.getOwner());
        it = sourceOps
                 .try_emplace(v, b.create<SourceOp>(v.getLoc(), v).getResult())
                 .first;
      }
      return it->second;
    };

    assert(stage && "Stage not found");
    Operation *stageTerminator = stage->getTerminator();
    for (auto *op : ops) {
      op->moveBefore(stageTerminator);

      // If the operation references values defined outside of this stage,
      // modify their uses to point to the corresponding SourceOp.
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.get().getParentBlock() != stage)
          operand.set(getOrCreateSourceOp(operand));
      }
    }
  }

  // Remove the unscheduled pipeline
  pipeline.replaceAllUsesWith(schedPipeline);
  pipeline.erase();
  return success();
}

void ScheduleLinearPipelinePass::runOnOperation() {
  for (auto &region : getOperation()->getRegions()) {
    for (auto pipeline :
         llvm::make_early_inc_range(region.getOps<UnscheduledPipelineOp>())) {
      if (failed(schedulePipeline(pipeline)))
        return signalPassFailure();
    }
  }
}

std::unique_ptr<mlir::Pass>
circt::pipeline::createScheduleLinearPipelinePass() {
  return std::make_unique<ScheduleLinearPipelinePass>();
}
