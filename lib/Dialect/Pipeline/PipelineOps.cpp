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

#include "circt/Dialect/Pipeline/Pipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace circt;
using namespace circt::pipeline;

#include "circt/Dialect/Pipeline/PipelineDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// UnscheduledPipelineOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyPipeline(PipelineLike op) {
  Block *entryStage = op.getEntryStage();
  if (entryStage->getNumArguments() != op.getInputs().size())
    return op.emitOpError("expected ")
           << op.getInputs().size()
           << " arguments in the pipeline body block, got "
           << entryStage->getNumArguments() << ".";

  for (size_t i = 0; i < op.getInputs().size(); i++) {
    Type expectedInArg = op.getInputs()[i].getType();
    Type bodyArg = entryStage->getArgument(i).getType();
    if (expectedInArg != bodyArg)
      return op.emitOpError("expected body block argument ")
             << i << " to have type " << expectedInArg << ", got " << bodyArg
             << ".";
  }

  return success();
}

LogicalResult UnscheduledPipelineOp::verify() { return verifyPipeline(*this); }

//===----------------------------------------------------------------------===//
// ScheduledPipelineOp
//===----------------------------------------------------------------------===//

void ScheduledPipelineOp::build(mlir::OpBuilder &odsBuilder,
                                mlir::OperationState &odsState,
                                ::mlir::TypeRange results,
                                mlir::ValueRange inputs, mlir::Value clock,
                                mlir::Value reset, mlir::Value stall) {
  odsState.addOperands(inputs);
  odsState.addOperands(clock);
  odsState.addOperands(reset);
  if (stall)
    odsState.addOperands(stall);
  auto *region = odsState.addRegion();
  odsState.addTypes(results);

  odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(inputs.size()), static_cast<int32_t>(1),
           static_cast<int32_t>(1), static_cast<int32_t>(stall ? 1 : 0)}));

  // Add the entry stage
  auto &entryBlock = region->emplaceBlock();
  llvm::SmallVector<Location> entryArgLocs(inputs.size(), odsState.location);
  entryBlock.addArguments(inputs.getTypes(), entryArgLocs);
}

Block *ScheduledPipelineOp::addStage() {
  OpBuilder builder(getContext());
  Block *stage = builder.createBlock(&getRegion());
  return stage;
}

llvm::SmallVector<Block *> ScheduledPipelineOp::getOrderedStages() {
  Block *currentStage = getEntryStage();
  llvm::SmallVector<Block *> orderedStages;
  do {
    orderedStages.push_back(currentStage);
    if (auto stageOp = dyn_cast<StageOp>(currentStage->getTerminator()))
      currentStage = stageOp.getNextStage();
    else
      currentStage = nullptr;
  } while (currentStage);

  return orderedStages;
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
  // input.
  return llvm::any_of(getStages(), [this](Block &block) {
    // The entry stage doesn't count since it'll always have arguments.
    if (&block == getEntryStage())
      return false;
    return block.getNumArguments() != 0;
  });
}

LogicalResult ScheduledPipelineOp::verify() {
  if (failed(verifyPipeline(*this)))
    return failure();

  // Phase invariant - if any block has arguments, we
  bool materialized = isMaterialized();
  if (materialized) {
    // Check that all values used within a stage are defined within the stage.
    for (auto &stage : getStages()) {
      for (auto &op : stage) {
        for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
          bool err = false;
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

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

SuccessorOperands StageOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  // Successor operands are everything but the "valid" input to this stage op.
  // Places a hard assumption on the regs and passthrough operands being next to
  // each other in the operand list.
  auto mutableRange =
      mlir::MutableOperandRange(getOperation(), 0, getNumOperands() - 1);
  return SuccessorOperands(mutableRange);
}

Block *StageOp::getSuccessorForOperands(ArrayRef<Attribute>) {
  return getNextStage();
}

LogicalResult StageOp::verify() {
  // Verify that the target block has the correct arguments as this stage op.
  llvm::SmallVector<Type> expectedTargetArgTypes;
  llvm::append_range(expectedTargetArgTypes, getRegisters().getTypes());
  llvm::append_range(expectedTargetArgTypes, getPassthroughs().getTypes());
  Block *targetStage = getNextStage();

  if (targetStage->getNumArguments() != expectedTargetArgTypes.size())
    return emitOpError("expected ") << expectedTargetArgTypes.size()
                                    << " arguments in the target stage, got "
                                    << targetStage->getNumArguments() << ".";

  for (auto [index, it] : llvm::enumerate(llvm::zip(
           expectedTargetArgTypes, targetStage->getArgumentTypes()))) {
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
    int64_t fromStage = stageMap[from];
    int64_t toStage = stageMap[to];
    return toStage - fromStage;
  };

  for (auto [i, res] : llvm::enumerate(getResults())) {
    for (auto &use : res.getUses()) {
      auto *user = use.getOwner();
      Block *userStage = user->getBlock();
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
