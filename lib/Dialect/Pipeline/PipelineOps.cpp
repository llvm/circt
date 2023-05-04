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

  // Check mixing of `pipeline.stage` and `pipeline.stage.register` ops.
  // This verifier thus ensures a proper phase ordering between stage ops
  // and their materialized stage register op counterparts.
  bool hasStageOps = !getOps<PipelineStageOp>().empty();
  bool hasStageRegOps = !getOps<PipelineStageRegisterOp>().empty();

  if (hasStageOps && hasStageRegOps)
    return emitOpError("mixing `pipeline.stage` and `pipeline.stage.register` "
                       "ops is illegal.");

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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  PipelineOp parent = getOperation()->getParentOfType<PipelineOp>();
  if (getOutputs().size() != parent.getResults().size())
    return emitOpError("expected ")
           << parent.getResults().size() << " return values, got "
           << getOutputs().size() << ".";

  bool isLatencyInsensitive = parent.isLatencyInsensitive();
  for (size_t i = 0; i < parent.getResults().size(); i++) {
    Type expectedType = parent.getResultTypes()[i];
    Type actualType = getOperandTypes()[i];
    if (isLatencyInsensitive)
      expectedType = expectedType.cast<esi::ChannelType>().getInner();
    if (expectedType != actualType)
      return emitOpError("expected argument ")
             << i << " to have type " << expectedType << ", got " << actualType
             << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PipelineStageRegisterOp
//===----------------------------------------------------------------------===//

void PipelineStageRegisterOp::build(OpBuilder &builder, OperationState &state,
                                    Value when, ValueRange regIns) {
  PipelineStageRegisterOp::build(builder, state, regIns.getTypes(), regIns,
                                 when);
  state.addTypes({when.getType()});
}

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"

void PipelineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"
      >();
}
