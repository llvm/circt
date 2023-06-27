//===- PipelineOps.h - Pipeline MLIR Operations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Pipeline ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Pipeline/PipelineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace circt;
using namespace circt::pipeline;

#include "circt/Dialect/Pipeline/PipelineDialect.cpp.inc"

#define DEBUG_TYPE "pipeline-ops"

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Block *block) {
  // Optional debug check - ensure that 'block' eventually leads to the
  // pipeline.
  LLVM_DEBUG({
    Operation *directParent = block->getParentOp();
    if (directParent != pipeline) {
      auto indirectParent =
          directParent->getParentOfType<ScheduledPipelineOp>();
      assert(indirectParent == pipeline && "block is not in the pipeline");
    }
  });

  while (block && block->getParent() != &pipeline.getRegion()) {
    // Go one level up.
    block = block->getParent()->getParentOp()->getBlock();
  }

  // This is a block within the pipeline region, so it must be a stage.
  return block;
}

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Operation *op) {
  return getParentStageInPipeline(pipeline, op->getBlock());
}

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Value v) {
  if (v.isa<BlockArgument>())
    return getParentStageInPipeline(pipeline,
                                    v.cast<BlockArgument>().getOwner());
  return getParentStageInPipeline(pipeline, v.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// UnscheduledPipelineOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyPipeline(PipelineLike op) {
  Block *entryStage = op.getEntryStage();
  llvm::SmallVector<Type> expectedInArgTypes;
  llvm::append_range(expectedInArgTypes, op.getInputs().getTypes());
  llvm::append_range(expectedInArgTypes, op.getExtInputs().getTypes());

  // Validate the stage data inputs.
  size_t nEntryStageArguments = entryStage->getArgumentTypes().size();
  if (nEntryStageArguments != expectedInArgTypes.size() + /*s0 valid*/ 1)
    return op.emitOpError("expected ")
           << expectedInArgTypes.size() + /*s0 valid*/ 1
           << " arguments in the pipeline entry block, got "
           << nEntryStageArguments << ".";

  TypeRange entryStageArguments =
      TypeRange(entryStage->getArgumentTypes()).drop_back();

  for (size_t i = 0; i < expectedInArgTypes.size(); i++) {
    Type expectedInArg = expectedInArgTypes[i];
    Type bodyArg = entryStageArguments[i];
    if (expectedInArg != bodyArg)
      return op.emitOpError("expected body block argument ")
             << i << " to have type " << expectedInArg << ", got " << bodyArg
             << ".";
  }

  // Verify that the entry stage is terminated with an i1 signal (valid signal).
  IntegerType lastEntryArgType =
      entryStage->getArguments().back().getType().dyn_cast<IntegerType>();
  if (!lastEntryArgType || !lastEntryArgType.isInteger(1))
    return op.emitOpError(
        "expected last argument to the entry stage to be an i1 "
        "signal (stage valid signal).");

  return success();
}

LogicalResult UnscheduledPipelineOp::verify() { return verifyPipeline(*this); }

//===----------------------------------------------------------------------===//
// ScheduledPipelineOp
//===----------------------------------------------------------------------===//

void ScheduledPipelineOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TypeRange results, ValueRange inputs,
                                ValueRange extInputs, Value clock, Value reset,
                                Value go, Value stall) {
  odsState.addOperands(inputs);
  odsState.addOperands(extInputs);
  odsState.addOperands(clock);
  odsState.addOperands(reset);
  odsState.addOperands(go);
  if (stall)
    odsState.addOperands(stall);

  odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(inputs.size()),
           static_cast<int32_t>(extInputs.size()), static_cast<int32_t>(1),
           static_cast<int32_t>(1), static_cast<int32_t>(1),
           static_cast<int32_t>(stall ? 1 : 0)}));

  auto *region = odsState.addRegion();
  odsState.addTypes(results);

  // Add the entry stage
  auto &entryBlock = region->emplaceBlock();
  llvm::SmallVector<Location> entryArgLocs(inputs.size(), odsState.location);
  entryBlock.addArguments(
      inputs.getTypes(),
      llvm::SmallVector<Location>(inputs.size(), odsState.location));
  entryBlock.addArguments(
      extInputs.getTypes(),
      llvm::SmallVector<Location>(extInputs.size(), odsState.location));
  // entry stage valid signal.
  entryBlock.addArgument(odsBuilder.getIntegerType(1), odsState.location);
}

Block *ScheduledPipelineOp::addStage() {
  OpBuilder builder(getContext());
  Block *stage = builder.createBlock(&getRegion());

  // Add the stage valid signal.
  stage->addArgument(builder.getIntegerType(1), getLoc());
  return stage;
}

void ScheduledPipelineOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  for (auto [i, block] : llvm::enumerate(getRegion())) {
    if (&block == getEntryStage()) {
      size_t nRegularInputs = getInputs().size();
      size_t nExternalInputs = getExtInputs().size();
      for (auto [argi, arg] :
           llvm::enumerate(block.getArguments().slice(0, nRegularInputs)))
        setNameFn(arg, llvm::formatv("s0_arg{0}", argi).str());

      for (auto [argi, arg] : llvm::enumerate(
               block.getArguments().slice(nRegularInputs, nExternalInputs)))
        setNameFn(arg, llvm::formatv("ext{0}", argi).str());
    } else {
      for (auto [argi, arg] : llvm::enumerate(block.getArguments().drop_back()))
        setNameFn(arg, llvm::formatv("s{0}_arg{1}", i, argi).str());
    }

    setNameFn(block.getArguments().back(),
              llvm::formatv("s{0}_valid", i).str());
  }
}
void ScheduledPipelineOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getDone(), "done");
}

// Implementation of getOrderedStages which also produces an error if
// there are any cfg cycles in the pipeline.
static FailureOr<llvm::SmallVector<Block *>>
getOrderedStagesFailable(ScheduledPipelineOp op) {
  llvm::DenseSet<Block *> visited;
  Block *currentStage = op.getEntryStage();
  llvm::SmallVector<Block *> orderedStages;
  do {
    if (!visited.insert(currentStage).second)
      return op.emitOpError("pipeline contains a cycle.");

    orderedStages.push_back(currentStage);
    if (auto stageOp = dyn_cast<StageOp>(currentStage->getTerminator()))
      currentStage = stageOp.getNextStage();
    else
      currentStage = nullptr;
  } while (currentStage);

  return {orderedStages};
}

llvm::SmallVector<Block *> ScheduledPipelineOp::getOrderedStages() {
  // Should always be safe, seeing as the pipeline itself has already been
  // verified.
  return *getOrderedStagesFailable(*this);
}

llvm::DenseMap<Block *, unsigned> ScheduledPipelineOp::getStageMap() {
  llvm::DenseMap<Block *, unsigned> stageMap;
  auto orderedStages = getOrderedStages();
  for (auto [index, stage] : llvm::enumerate(orderedStages))
    stageMap[stage] = index;

  return stageMap;
}

Block *ScheduledPipelineOp::getLastStage() { return getOrderedStages().back(); }

bool ScheduledPipelineOp::isMaterialized() {
  // We determine materialization as if any pipeline stage has an explicit
  // input (apart from the stage valid signal).
  return llvm::any_of(getStages(), [this](Block &block) {
    // The entry stage doesn't count since it'll always have arguments.
    if (&block == getEntryStage())
      return false;
    return block.getNumArguments() > 1;
  });
}

LogicalResult ScheduledPipelineOp::verify() {
  // Generic scheduled/unscheduled verification.
  if (failed(verifyPipeline(*this)))
    return failure();

  // Verify that all block are terminated properly.
  auto &stages = getStages();
  for (Block &stage : stages) {
    if (stage.empty() || !isa<ReturnOp, StageOp>(stage.back()))
      return emitOpError("all blocks must be terminated with a "
                         "`pipeline.stage` or `pipeline.return` op.");
  }

  if (failed(getOrderedStagesFailable(*this)))
    return failure();

  // Verify that every stage has a stage valid block argument.
  for (auto [i, block] : llvm::enumerate(stages)) {
    bool err = true;
    if (block.getNumArguments() != 0) {
      auto lastArgType =
          block.getArguments().back().getType().dyn_cast<IntegerType>();
      err = !lastArgType || lastArgType.getWidth() != 1;
    }
    if (err)
      return emitOpError("block " + std::to_string(i) +
                         " must have an i1 argument as the last block argument "
                         "(stage valid signal).");
  }

  // Cache external inputs in a set for fast lookup.
  llvm::DenseSet<Value> extInputs;
  for (auto extInput : getInnerExtInputs())
    extInputs.insert(extInput);

  // Phase invariant - if any block has arguments apart from the stage valid
  // argument, we are in register materialized mode. Check that all values used
  // within a stage are defined within the stage.
  bool materialized = isMaterialized();
  if (materialized) {
    for (auto &stage : stages) {
      for (auto &op : stage) {
        for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
          bool err = false;
          if (extInputs.contains(operand)) {
            // This is an external input; legal to reference everywhere.
            continue;
          }

          if (auto *definingOp = operand.getDefiningOp()) {
            // Constants are allowed to be used across stages.
            if (definingOp->hasTrait<OpTrait::ConstantLike>())
              continue;
            err = definingOp->getBlock() != &stage;
          } else {
            // This is a block argument;
            err = !llvm::is_contained(stage.getArguments(), operand);
          }

          if (err)
            return op.emitOpError(
                       "Pipeline is in register materialized mode - operand ")
                   << index
                   << " is defined in a different stage, which is illegal.";
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  size_t nInputs = getInputs().size();
  auto expectedResults = TypeRange(parent->getResultTypes()).drop_back();
  size_t expectedNResults = expectedResults.size();
  if (nInputs != expectedNResults)
    return emitOpError("expected ")
           << expectedNResults << " return values, got " << nInputs << ".";

  for (auto [inType, reqType] :
       llvm::zip(getInputs().getTypes(), expectedResults)) {
    if (inType != reqType)
      return emitOpError("expected return value of type ")
             << reqType << ", got " << inType << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

LogicalResult StageOp::verify() {
  // Verify that the target block has the correct arguments as this stage op.
  llvm::SmallVector<Type> expectedTargetArgTypes;
  llvm::append_range(expectedTargetArgTypes, getRegisters().getTypes());
  llvm::append_range(expectedTargetArgTypes, getPassthroughs().getTypes());
  Block *targetStage = getNextStage();
  // Expected types is everything but the stage valid signal.
  TypeRange targetStageArgTypes =
      TypeRange(targetStage->getArgumentTypes()).drop_back();

  if (targetStageArgTypes.size() != expectedTargetArgTypes.size())
    return emitOpError("expected ") << expectedTargetArgTypes.size()
                                    << " arguments in the target stage, got "
                                    << targetStageArgTypes.size() << ".";

  for (auto [index, it] : llvm::enumerate(
           llvm::zip(expectedTargetArgTypes, targetStageArgTypes))) {
    auto [arg, barg] = it;
    if (arg != barg)
      return emitOpError("expected target stage argument ")
             << index << " to have type " << arg << ", got " << barg << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LatencyOp
//===----------------------------------------------------------------------===//

LogicalResult LatencyOp::verify() {
  ScheduledPipelineOp scheduledPipelineParent =
      dyn_cast<ScheduledPipelineOp>(getOperation()->getParentOp());

  if (!scheduledPipelineParent) {
    // Nothing to verify, got to assume that anything goes in an unscheduled
    // pipeline.
    return success();
  }

  // Verify that the resulting values aren't referenced before they are
  // accessible.
  size_t latency = getLatency();
  Block *definingStage = getOperation()->getBlock();

  llvm::DenseMap<Block *, unsigned> stageMap =
      scheduledPipelineParent.getStageMap();

  auto stageDistance = [&](Block *from, Block *to) {
    assert(stageMap.count(from) && "stage 'from' not contained in pipeline");
    assert(stageMap.count(to) && "stage 'to' not contained in pipeline");
    int64_t fromStage = stageMap[from];
    int64_t toStage = stageMap[to];
    return toStage - fromStage;
  };

  for (auto [i, res] : llvm::enumerate(getResults())) {
    for (auto &use : res.getUses()) {
      auto *user = use.getOwner();

      // The user may reside within a block which is not a stage (e.g. inside
      // a pipeline.latency op). Determine the stage which this use resides
      // within.
      Block *userStage =
          getParentStageInPipeline(scheduledPipelineParent, user);
      unsigned useDistance = stageDistance(definingStage, userStage);

      // Is this a stage op and is the value passed through? if so, this is a
      // legal use.
      StageOp stageOp = dyn_cast<StageOp>(user);
      if (userStage == definingStage && stageOp) {
        if (llvm::is_contained(stageOp.getPassthroughs(), res))
          continue;
      }

      // The use is not a passthrough. Check that the distance between
      // the defining stage and the user stage is at least the latency of the
      // result.
      if (useDistance < latency) {
        auto diag = emitOpError("result ")
                    << i << " is used before it is available.";
        diag.attachNote(user->getLoc())
            << "use was operand " << use.getOperandNumber()
            << ". The result is available " << latency - useDistance
            << " stages later than this use.";
        return diag;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LatencyReturnOp
//===----------------------------------------------------------------------===//

LogicalResult LatencyReturnOp::verify() {
  LatencyOp parent = cast<LatencyOp>(getOperation()->getParentOp());
  size_t nInputs = getInputs().size();
  size_t nResults = parent->getNumResults();
  if (nInputs != nResults)
    return emitOpError("expected ")
           << nResults << " return values, got " << nInputs << ".";

  for (auto [inType, reqType] :
       llvm::zip(getInputs().getTypes(), parent->getResultTypes())) {
    if (inType != reqType)
      return emitOpError("expected return value of type ")
             << reqType << ", got " << inType << ".";
  }

  return success();
}

#include "circt/Dialect/Pipeline/PipelineInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"

void PipelineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"
      >();
}
