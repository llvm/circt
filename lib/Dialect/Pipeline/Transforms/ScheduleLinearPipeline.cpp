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
#include "circt/Dialect/SSP/SSPAttributes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <type_traits>

#define DEBUG_TYPE "pipeline-schedule-linear"

namespace circt {
namespace pipeline {
#define GEN_PASS_DECL_SCHEDULELINEARPIPELINE
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
  template <typename ProblemTy>
  LogicalResult
  schedulePipeline(UnscheduledPipelineOp pipeline,
                   std::optional<float> cycleTime = std::nullopt,
                   std::optional<int> initiationInterval = std::nullopt);
};

} // end anonymous namespace

// Returns true if 'op' should be ignored in the scheduling problem.
static bool ignoreOp(Operation *op) {
  return op->hasTrait<OpTrait::ConstantLike>();
}

template <typename ProblemTy>
LogicalResult ScheduleLinearPipelinePass::schedulePipeline(
    UnscheduledPipelineOp pipeline, std::optional<float> cycleTime,
    std::optional<int> initiationInterval) {
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
  ProblemTy problem(pipeline);

  DenseMap<SymbolRefAttr, typename ProblemTy::OperatorType> operatorTypes;
  SmallDenseMap<StringAttr, unsigned> oprIds;

  // Set operation operator types.
  auto returnOp =
      cast<pipeline::ReturnOp>(pipeline.getEntryStage()->getTerminator());
  for (auto &op : pipeline.getOps()) {
    // Skip if is a known non-functional operator
    if (ignoreOp(&op))
      continue;

    typename ProblemTy::OperatorType operatorType;
    bool isReturnOp = &op == returnOp.getOperation();
    if (isReturnOp) {
      // Construct an operator type for the return op (not an externally defined
      // operator type since it is intrinsic to this pass).
      operatorType = problem.getOrInsertOperatorType("return");
      problem.setLatency(operatorType, 0);

      // set delay to 0 for return op in chaining related problems
      if constexpr (std::is_same_v<ProblemTy, ChainingProblem> ||
                    std::is_same_v<ProblemTy, ChainingCyclicProblem>) {
        problem.setIncomingDelay(operatorType, 0);
        problem.setOutgoingDelay(operatorType, 0);
      }
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

        if constexpr (std::is_same_v<ProblemTy, Problem> ||
                      std::is_same_v<ProblemTy, CyclicProblem>) {
          auto insertRes = operatorTypes.try_emplace(
              operatorTypeAttr,
              ssp::loadOperatorType<ProblemTy, ssp::LatencyAttr>(
                  problem, opTypeOp, oprIds));
          operatorTypeIt = insertRes.first;
        }

        else if constexpr (std::is_same_v<ProblemTy, SharedOperatorsProblem> ||
                           std::is_same_v<ProblemTy, ModuloProblem>) {
          auto insertRes = operatorTypes.try_emplace(
              operatorTypeAttr,
              ssp::loadOperatorType<ProblemTy, ssp::LatencyAttr,
                                    ssp::LimitAttr>(problem, opTypeOp, oprIds));
          operatorTypeIt = insertRes.first;
        }

        else if constexpr (std::is_same_v<ProblemTy, ChainingProblem> ||
                           std::is_same_v<ProblemTy, ChainingCyclicProblem>) {
          auto insertRes = operatorTypes.try_emplace(
              operatorTypeAttr,
              ssp::loadOperatorType<ProblemTy, ssp::LatencyAttr,
                                    ssp::IncomingDelayAttr,
                                    ssp::OutgoingDelayAttr>(problem, opTypeOp,
                                                            oprIds));
          operatorTypeIt = insertRes.first;
        }
      }
      operatorType = operatorTypeIt->second;
    }

    // set initiation interval for relevant problems
    if constexpr (std::is_same_v<ProblemTy, CyclicProblem> ||
                  std::is_same_v<ProblemTy, ModuloProblem> ||
                  std::is_same_v<ProblemTy, ChainingCyclicProblem>) {
      problem.setInitiationInterval(initiationInterval.value());
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

  LogicalResult result = failure();
  if constexpr (std::is_same_v<ProblemTy, Problem> ||
                std::is_same_v<ProblemTy, CyclicProblem> ||
                std::is_same_v<ProblemTy, SharedOperatorsProblem> ||
                std::is_same_v<ProblemTy, ModuloProblem>) {
    result = scheduling::scheduleSimplex(problem, returnOp.getOperation());
  }
  if constexpr (std::is_same_v<ProblemTy, ChainingProblem> ||
                std::is_same_v<ProblemTy, ChainingCyclicProblem>) {
    result = scheduling::scheduleSimplex(problem, returnOp.getOperation(),
                                         cycleTime.value());
  }

  if (failed(result))
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
    assert(stage && "Stage not found");
    Operation *stageTerminator = stage->getTerminator();
    for (auto *op : ops)
      op->moveBefore(stageTerminator);
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

      if (this->problemType == "base") {
        if (failed(schedulePipeline<Problem>(pipeline)))
          return signalPassFailure();
      } else if (this->problemType == "cyclic") {
        if (failed(schedulePipeline<CyclicProblem>(pipeline, std::nullopt,
                                                   this->initInterval)))
          return signalPassFailure();
      } else if (this->problemType == "shared_operators") {
        if (failed(schedulePipeline<SharedOperatorsProblem>(pipeline)))
          return signalPassFailure();
      } else if (this->problemType == "modulo") {
        if (failed(schedulePipeline<ModuloProblem>(pipeline, std::nullopt,
                                                   this->initInterval)))
          return signalPassFailure();
      } else if (this->problemType == "chaining") {
        if (failed(
                schedulePipeline<ChainingProblem>(pipeline, this->cycleTime)))
          return signalPassFailure();
      } else if (this->problemType == "cyclic_chaining") {
        if (failed(schedulePipeline<ChainingCyclicProblem>(pipeline,
                                                           this->cycleTime)))
          return signalPassFailure();
      } else {
        return signalPassFailure();
      }
    }
  }
}

std::unique_ptr<mlir::Pass>
circt::pipeline::createScheduleLinearPipelinePass() {
  return std::make_unique<ScheduleLinearPipelinePass>();
}
