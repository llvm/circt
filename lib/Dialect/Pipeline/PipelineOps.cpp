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

#include "circt/Dialect/ESI/ESITypes.h"
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
// PipelineOp
//===----------------------------------------------------------------------===//

LogicalResult PipelineOp::verify() {
  bool anyInputIsAChannel = llvm::any_of(getInputs(), [](Value operand) {
    return operand.getType().isa<esi::ChannelType>();
  });
  bool anyOutputIsAChannel = llvm::any_of(
      getResultTypes(), [](Type type) { return type.isa<esi::ChannelType>(); });

  if ((anyInputIsAChannel || anyOutputIsAChannel) && !isLatencyInsensitive()) {
    return emitOpError("if any port of this pipeline is an ESI channel, all "
                       "ports must be ESI channels.");
  }

  if (getBodyBlock()->getNumArguments() != getInputs().size())
    return emitOpError("expected ")
           << getInputs().size()
           << " arguments in the pipeline body block, got "
           << getBodyBlock()->getNumArguments() << ".";

  for (size_t i = 0; i < getInputs().size(); i++) {
    Type expectedInArg = getInputs()[i].getType();
    Type bodyArg = getBodyBlock()->getArgument(i).getType();

    if (isLatencyInsensitive())
      expectedInArg = expectedInArg.cast<esi::ChannelType>().getInner();

    if (expectedInArg != bodyArg)
      return emitOpError("expected body block argument ")
             << i << " to have type " << expectedInArg << ", got " << bodyArg
             << ".";
  }

  // Check mixing of stage-like operations. These are not allowed to coexist.
  bool hasDelimiterOps = !getOps<StageSeparatingOp>().empty();
  bool hasDelimiterRegsOps = !getOps<StageSeparatingRegOp>().empty();
  bool hasStageOps = !getOps<StageOp>().empty();
  size_t phaseKinds = hasDelimiterOps + hasDelimiterRegsOps + hasStageOps;
  if (phaseKinds > 1)
    return emitOpError("pipeline contains a mix of stage-like operations.");

  return success();
}

bool PipelineOp::isLatencyInsensitive() {
  bool allInputsAreChannels = llvm::all_of(getInputs(), [](Value operand) {
    return operand.getType().isa<esi::ChannelType>();
  });
  bool allOutputsAreChannels = llvm::all_of(
      getResultTypes(), [](Type type) { return type.isa<esi::ChannelType>(); });
  return allInputsAreChannels && allOutputsAreChannels;
}

// ===----------------------------------------------------------------------===//
// StageSeparatingOp
// ===----------------------------------------------------------------------===//

// Returns the index of this stage in the pipeline.
unsigned StageSeparatingOp::index() {
  auto stageOps =
      getOperation()->getParentOfType<PipelineOp>().getOps<StageSeparatingOp>();
  return std::distance(stageOps.begin(), llvm::find(stageOps, *this));
}

// ===----------------------------------------------------------------------===//
// StageSeparatingRegOp
// ===----------------------------------------------------------------------===//

unsigned StageSeparatingRegOp::index() {
  auto stageOps = getOperation()
                      ->getParentOfType<PipelineOp>()
                      .getOps<StageSeparatingRegOp>();
  return std::distance(stageOps.begin(), llvm::find(stageOps, *this));
}

void StageSeparatingRegOp::build(OpBuilder &builder, OperationState &state,
                                 Value enable, ValueRange inputs) {
  StageSeparatingRegOp::build(builder, state, inputs.getTypes(), inputs,
                              enable);
  state.addTypes({enable.getType()});
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  PipelineOp parent = getOperation()->getParentOfType<PipelineOp>();
  if (getInputs().size() != parent.getResults().size())
    return emitOpError("expected ")
           << parent.getResults().size() << " return values, got "
           << getInputs().size() << ".";

  bool isLatencyInsensitive = parent.isLatencyInsensitive();
  for (size_t i = 0; i < parent.getResults().size(); i++) {
    Type expectedType = parent.getResultTypes()[i];
    Type actualType = getOperandTypes()[i];
    if (isLatencyInsensitive) {
      expectedType = expectedType.dyn_cast<esi::ChannelType>().getInner();
      if (!expectedType)
        return emitOpError("expected ESI channel type, got ")
               << parent.getResultTypes()[i] << ".";
    }
    if (expectedType != actualType)
      return emitOpError("expected argument ")
             << i << " to have type " << expectedType << ", got " << actualType
             << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

void StageOp::build(OpBuilder &builder, OperationState &odsState, Value enable,
                    ValueRange inputs, TypeRange outputs) {
  odsState.addOperands(inputs);
  odsState.addOperands(enable);
  odsState.addTypes(outputs);
  // Valid output
  odsState.addTypes(builder.getI1Type());

  // Build inner region.
  Region *region = odsState.addRegion();
  Block *bodyBlock = new Block();
  region->push_back(bodyBlock);
  bodyBlock->addArguments(
      inputs.getTypes(),
      SmallVector<Location>(inputs.size(), odsState.location));
  bodyBlock->addArguments({enable.getType()},
                          SmallVector<Location>(1, odsState.location));
  StageOp::ensureTerminator(*region, builder, odsState.location);
}

LogicalResult StageOp::verify() {
  auto inputs = getInputs();
  if (getBodyBlock()->getNumArguments() != (inputs.size() + /*valid*/ 1))
    return emitOpError("expected ")
           << inputs.size() + 1
           << " arguments in the pipeline stage body block, got "
           << getBodyBlock()->getNumArguments() << ".";

  for (size_t i = 0; i < inputs.size(); i++) {
    Type expectedInArg = inputs[i].getType();
    Type bodyArg = getBodyBlock()->getArgument(i).getType();
    if (expectedInArg != bodyArg)
      return emitOpError("expected stage body block argument ")
             << i << " to have type " << expectedInArg << ", got " << bodyArg
             << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageReturnOp
//===----------------------------------------------------------------------===//

LogicalResult StageReturnOp::verify() {
  StageOp parent = getOperation()->getParentOfType<StageOp>();
  size_t nRetVals = getRegs().size() + getPassthroughs().size();
  if (nRetVals != parent.getOutputs().size())
    return emitOpError("expected ")
           << parent.getOutputs().size() << " return values, got " << nRetVals
           << ".";

  llvm::SmallVector<Type> retTypes;
  llvm::append_range(retTypes, getRegs().getTypes());
  llvm::append_range(retTypes, getPassthroughs().getTypes());
  for (size_t i = 0; i < parent.getOutputs().size(); i++) {
    Type expectedType = parent.getOutputs().getTypes()[i];
    Type actualType = retTypes[i];
    if (expectedType != actualType)
      return emitOpError("expected argument ")
             << i << " to have type " << expectedType << ", got " << actualType
             << ".";
  }

  return success();
}

void StageReturnOp::setOperands(Value valid, ValueRange regs,
                                ValueRange passthroughs) {
  getOperation()->insertOperands(0, regs);
  getOperation()->insertOperands(regs.size(), passthroughs);
  getOperation()->insertOperands(regs.size() + passthroughs.size(), {valid});
  llvm::SmallVector<int32_t, 3> operandSizes;
  operandSizes.push_back(regs.size());
  operandSizes.push_back(passthroughs.size());
  operandSizes.push_back(1);
  getOperation()->setAttr(
      "operand_segment_sizes",
      mlir::DenseI32ArrayAttr::get(getContext(), operandSizes));
}

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"

void PipelineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"
      >();
}
