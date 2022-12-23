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
// PipelineWhileOp
//===----------------------------------------------------------------------===//

ParseResult PipelineWhileOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  // Parse initiation interval.
  IntegerAttr ii;
  if (parser.parseKeyword("II") || parser.parseEqual() ||
      parser.parseAttribute(ii))
    return failure();
  result.addAttribute("II", ii);

  // Parse optional trip count.
  if (succeeded(parser.parseOptionalKeyword("trip_count"))) {
    IntegerAttr tripCount;
    if (parser.parseEqual() || parser.parseAttribute(tripCount))
      return failure();
    result.addAttribute("tripCount", tripCount);
  }

  // Parse iter_args assignment list.
  SmallVector<OpAsmParser::Argument> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseAssignmentList(regionArgs, operands))
      return failure();
  }

  // Parse function type from iter_args to results.
  FunctionType type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();

  // Function result type is the pipeline result type.
  result.addTypes(type.getResults());

  // Resolve iter_args operands.
  for (auto [regionArg, operand, type] :
       llvm::zip(regionArgs, operands, type.getInputs())) {
    regionArg.type = type;
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }

  // Parse condition region.
  Region *condition = result.addRegion();
  if (parser.parseRegion(*condition, regionArgs))
    return failure();

  // Parse stages region.
  if (parser.parseKeyword("do"))
    return failure();
  Region *stages = result.addRegion();
  if (parser.parseRegion(*stages, regionArgs))
    return failure();

  return success();
}

void PipelineWhileOp::print(OpAsmPrinter &p) {
  // Print the initiation interval.
  p << " II = " << ' ' << getII();

  // Print the optional tripCount.
  if (getTripCount())
    p << " trip_count = " << ' ' << *getTripCount();

  // Print iter_args assignment list.
  p << " iter_args(";
  llvm::interleaveComma(
      llvm::zip(getStages().getArguments(), getIterArgs()), p,
      [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ") : ";

  // Print function type from iter_args to results.
  auto type = FunctionType::get(getContext(), getStages().getArgumentTypes(),
                                getResultTypes());
  p.printType(type);

  // Print condition region.
  p << ' ';
  p.printRegion(getCondition(), /*printEntryBlockArgs=*/false);
  p << " do";

  // Print stages region.
  p << ' ';
  p.printRegion(getStages(), /*printEntryBlockArgs=*/false);
}

LogicalResult PipelineWhileOp::verify() {
  // Verify the condition block is "combinational" based on an allowlist of
  // Arithmetic ops.
  Block &conditionBlock = getCondition().front();
  Operation *nonCombinational;
  WalkResult conditionWalk = conditionBlock.walk([&](Operation *op) {
    if (isa<PipelineDialect>(op->getDialect()))
      return WalkResult::advance();

    if (!isa<arith::AddIOp, arith::AndIOp, arith::BitcastOp, arith::CmpIOp,
             arith::ConstantOp, arith::IndexCastOp, arith::MulIOp, arith::OrIOp,
             arith::SelectOp, arith::ShLIOp, arith::ExtSIOp, arith::CeilDivSIOp,
             arith::DivSIOp, arith::FloorDivSIOp, arith::RemSIOp,
             arith::ShRSIOp, arith::SubIOp, arith::TruncIOp, arith::DivUIOp,
             arith::RemUIOp, arith::ShRUIOp, arith::XOrIOp, arith::ExtUIOp>(
            op)) {
      nonCombinational = op;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (conditionWalk.wasInterrupted())
    return emitOpError("condition must have a combinational body, found ")
           << *nonCombinational;

  // Verify the condition block terminates with a value of type i1.
  TypeRange conditionResults =
      conditionBlock.getTerminator()->getOperandTypes();
  if (conditionResults.size() != 1)
    return emitOpError("condition must terminate with a single result, found ")
           << conditionResults;

  if (conditionResults.front() != IntegerType::get(getContext(), 1))
    return emitOpError("condition must terminate with an i1 result, found ")
           << conditionResults.front();

  // Verify the stages block contains at least one stage and a terminator.
  Block &stagesBlock = getStages().front();
  if (stagesBlock.getOperations().size() < 2)
    return emitOpError("stages must contain at least one stage");

  int64_t lastStartTime = -1;
  for (Operation &inner : stagesBlock) {
    // Verify the stages block contains only `pipeline.while.stage` and
    // `pipeline.terminator` ops.
    if (!isa<PipelineWhileStageOp, PipelineTerminatorOp>(inner))
      return emitOpError("stages may only contain 'pipeline.while.stage' or "
                         "'pipeline.terminator' ops, found ")
             << inner;

    // Verify the stage start times are monotonically increasing.
    if (auto stage = dyn_cast<PipelineWhileStageOp>(inner)) {
      if (lastStartTime == -1) {
        lastStartTime = stage.getStart();
        continue;
      }

      if (lastStartTime >= stage.getStart())
        return stage.emitOpError("'start' must be after previous 'start' (")
               << lastStartTime << ')';

      lastStartTime = stage.getStart();
    }
  }

  return success();
}

void PipelineWhileOp::build(OpBuilder &builder, OperationState &state,
                            TypeRange resultTypes, IntegerAttr ii,
                            std::optional<IntegerAttr> tripCount,
                            ValueRange iterArgs) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("II", ii);
  if (tripCount)
    state.addAttribute("tripCount", *tripCount);
  state.addOperands(iterArgs);

  Region *condRegion = state.addRegion();
  Block &condBlock = condRegion->emplaceBlock();

  SmallVector<Location, 4> argLocs;
  for (auto arg : iterArgs)
    argLocs.push_back(arg.getLoc());
  condBlock.addArguments(iterArgs.getTypes(), argLocs);
  builder.setInsertionPointToEnd(&condBlock);
  builder.create<PipelineRegisterOp>(builder.getUnknownLoc(), ValueRange());

  Region *stagesRegion = state.addRegion();
  Block &stagesBlock = stagesRegion->emplaceBlock();
  stagesBlock.addArguments(iterArgs.getTypes(), argLocs);
  builder.setInsertionPointToEnd(&stagesBlock);
  builder.create<PipelineTerminatorOp>(builder.getUnknownLoc(), ValueRange(),
                                       ValueRange());
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

//===----------------------------------------------------------------------===//
// PipelineWhileStageOp
//===----------------------------------------------------------------------===//

LogicalResult PipelineWhileStageOp::verify() {
  if (getStart() < 0)
    return emitOpError("'start' must be non-negative");

  return success();
}

void PipelineWhileStageOp::build(OpBuilder &builder, OperationState &state,
                                 TypeRange resultTypes, IntegerAttr start) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("start", start);

  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  builder.create<PipelineRegisterOp>(builder.getUnknownLoc(), ValueRange());
}

unsigned PipelineWhileStageOp::getStageNumber() {
  unsigned number = 0;
  auto *op = getOperation();
  auto parent = op->getParentOfType<PipelineWhileOp>();
  Operation *stage = &parent.getStagesBlock().front();
  while (stage != op && stage->getNextNode()) {
    ++number;
    stage = stage->getNextNode();
  }
  return number;
}

//===----------------------------------------------------------------------===//
// PipelineRegisterOp
//===----------------------------------------------------------------------===//

LogicalResult PipelineRegisterOp::verify() {
  PipelineWhileStageOp stage = (*this)->getParentOfType<PipelineWhileStageOp>();

  // If this doesn't terminate a stage, it is terminating the condition.
  if (stage == nullptr)
    return success();

  // Verify stage terminates with the same types as the result types.
  TypeRange registerTypes = getOperandTypes();
  TypeRange resultTypes = stage.getResultTypes();
  if (registerTypes != resultTypes)
    return emitOpError("operand types (")
           << registerTypes << ") must match result types (" << resultTypes
           << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// PipelineTerminatorOp
//===----------------------------------------------------------------------===//

LogicalResult PipelineTerminatorOp::verify() {
  PipelineWhileOp pipeline = (*this)->getParentOfType<PipelineWhileOp>();

  // Verify pipeline terminates with the same `iter_args` types as the pipeline.
  auto iterArgs = getIterArgs();
  TypeRange terminatorArgTypes = iterArgs.getTypes();
  TypeRange pipelineArgTypes = pipeline.getIterArgs().getTypes();
  if (terminatorArgTypes != pipelineArgTypes)
    return emitOpError("'iter_args' types (")
           << terminatorArgTypes << ") must match pipeline 'iter_args' types ("
           << pipelineArgTypes << ")";

  // Verify `iter_args` are defined by a pipeline stage.
  for (auto iterArg : iterArgs)
    if (iterArg.getDefiningOp<PipelineWhileStageOp>() == nullptr)
      return emitOpError(
          "'iter_args' must be defined by a 'pipeline.while.stage'");

  // Verify pipeline terminates with the same result types as the pipeline.
  auto opResults = getResults();
  TypeRange terminatorResultTypes = opResults.getTypes();
  TypeRange pipelineResultTypes = pipeline.getResultTypes();
  if (terminatorResultTypes != pipelineResultTypes)
    return emitOpError("'results' types (")
           << terminatorResultTypes << ") must match pipeline result types ("
           << pipelineResultTypes << ")";

  // Verify `results` are defined by a pipeline stage.
  for (auto result : opResults)
    if (result.getDefiningOp<PipelineWhileStageOp>() == nullptr)
      return emitOpError(
          "'results' must be defined by a 'pipeline.while.stage'");

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"

void PipelineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"
      >();
}
