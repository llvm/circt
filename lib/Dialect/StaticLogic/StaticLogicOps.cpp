//===- StaticLogicOps.h - StaticLogic MLIR Operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the StaticLogic ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace circt;
using namespace circt::staticlogic;

#include "circt/Dialect/StaticLogic/StaticLogicDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PipelineWhileOp
//===----------------------------------------------------------------------===//

static ParseResult parsePipelineWhileOp(OpAsmParser &parser,
                                        OperationState &result) {
  // Parse initiation interval.
  IntegerAttr ii;
  if (parser.parseKeyword("II") || parser.parseEqual() ||
      parser.parseAttribute(ii))
    return failure();
  result.addAttribute("II", ii);

  // Parse iter_args assignment list.
  SmallVector<OpAsmParser::OperandType> regionArgs, operands;
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
  for (auto [operand, type] : llvm::zip(operands, type.getInputs()))
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();

  // Parse condition region.
  Region *condition = result.addRegion();
  parser.parseRegion(*condition, regionArgs, type.getInputs());

  // Parse stages region.
  if (parser.parseKeyword("do"))
    return failure();
  Region *stages = result.addRegion();
  parser.parseRegion(*stages, regionArgs, type.getInputs());

  return success();
}

static void printPipelineWhileOp(OpAsmPrinter &p, PipelineWhileOp op) {
  // Print the initiation interval.
  p << " II = " << ' ' << op.II();

  // Print iter_args assignment list.
  p << " iter_args(";
  llvm::interleaveComma(
      llvm::zip(op.stages().getArguments(), op.iterArgs()), p,
      [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ") : ";

  // Print function type from iter_args to results.
  auto type = FunctionType::get(op.getContext(), op.stages().getArgumentTypes(),
                                op.getResultTypes());
  p.printType(type);

  // Print condition region.
  p << ' ';
  p.printRegion(op.condition(), /*printEntryBlockArgs=*/false);
  p << " do";

  // Print stages region.
  p << ' ';
  p.printRegion(op.stages(), /*printEntryBlockArgs=*/false);
}

static LogicalResult verifyPipelineWhileOp(PipelineWhileOp op) {
  // Verify the condition block is "combinational" based on an allowlist of
  // Arithmetic ops.
  Block &conditionBlock = op.condition().front();
  Operation *nonCombinational;
  WalkResult conditionWalk = conditionBlock.walk([&](Operation *op) {
    if (isa<StaticLogicDialect>(op->getDialect()))
      return WalkResult::advance();

    if (!isa<arith::AddIOp, arith::AndIOp, arith::BitcastOp, arith::CmpIOp,
             arith::ConstantOp, arith::IndexCastOp, arith::MulIOp, arith::OrIOp,
             SelectOp, arith::ShLIOp, arith::ExtSIOp, arith::CeilDivSIOp,
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
    return op.emitOpError("condition must have a combinational body, found ")
           << *nonCombinational;

  // Verify the condition block terminates with a value of type i1.
  TypeRange conditionResults =
      conditionBlock.getTerminator()->getOperandTypes();
  if (conditionResults.size() != 1)
    return op.emitOpError(
               "condition must terminate with a single result, found ")
           << conditionResults;

  if (conditionResults.front() != IntegerType::get(op.getContext(), 1))
    return op.emitOpError("condition must terminate with an i1 result, found ")
           << conditionResults.front();

  // Verify the stages block contains at least one stage and a terminator.
  Block &stagesBlock = op.stages().front();
  if (stagesBlock.getOperations().size() < 2)
    return op.emitOpError("stages must contain at least one stage");

  int64_t lastStartTime = -1;
  for (Operation &inner : stagesBlock) {
    // Verify the stages block contains only `staticlogic.pipeline.stage` and
    // `staticlogic.pipeline.terminator` ops.
    if (!isa<PipelineStageOp, PipelineTerminatorOp>(inner))
      return op.emitOpError(
                 "stages may only contain 'staticlogic.pipeline.stage' or "
                 "'staticlogic.pipeline.terminator' ops, found ")
             << inner;

    // Verify the stage start times are monotonically increasing.
    if (auto stage = dyn_cast<PipelineStageOp>(inner)) {
      if (lastStartTime == -1) {
        lastStartTime = stage.start();
        continue;
      }

      if (lastStartTime >= stage.start())
        return stage.emitOpError("'start' must be after previous 'start' (")
               << lastStartTime << ')';

      lastStartTime = stage.start();
    }
  }

  return success();
}

void PipelineWhileOp::build(OpBuilder &builder, OperationState &state,
                            TypeRange resultTypes, IntegerAttr ii,
                            ValueRange iterArgs) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("II", ii);
  state.addOperands(iterArgs);

  Region *condRegion = state.addRegion();
  Block &condBlock = condRegion->emplaceBlock();
  condBlock.addArguments(iterArgs.getTypes());
  builder.setInsertionPointToEnd(&condBlock);
  builder.create<PipelineRegisterOp>(builder.getUnknownLoc(), ValueRange());

  Region *stagesRegion = state.addRegion();
  Block &stagesBlock = stagesRegion->emplaceBlock();
  stagesBlock.addArguments(iterArgs.getTypes());
  builder.setInsertionPointToEnd(&stagesBlock);
  builder.create<PipelineTerminatorOp>(builder.getUnknownLoc(), ValueRange(),
                                       ValueRange());
}

//===----------------------------------------------------------------------===//
// PipelineStageOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyPipelineStageOp(PipelineStageOp op) {
  if (op.start() < 0)
    return op.emitOpError("'start' must be non-negative");

  return success();
}

void PipelineStageOp::build(OpBuilder &builder, OperationState &state,
                            TypeRange resultTypes, IntegerAttr start) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("start", start);

  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  builder.create<PipelineRegisterOp>(builder.getUnknownLoc(), ValueRange());
}

//===----------------------------------------------------------------------===//
// PipelineRegisterOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyPipelineRegisterOp(PipelineRegisterOp op) {
  PipelineStageOp stage = op->getParentOfType<PipelineStageOp>();

  // If this doesn't terminate a stage, it is terminating the condition.
  if (stage == nullptr)
    return success();

  // Verify stage terminates with the same types as the result types.
  TypeRange registerTypes = op.getOperandTypes();
  TypeRange resultTypes = stage.getResultTypes();
  if (registerTypes != resultTypes)
    return op.emitOpError("operand types (")
           << registerTypes << ") must match result types (" << resultTypes
           << ")";

  return success();
}

unsigned PipelineStageOp::getStageNumber() {
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
// PipelineTerminatorOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyPipelineTerminatorOp(PipelineTerminatorOp op) {
  PipelineWhileOp pipeline = op->getParentOfType<PipelineWhileOp>();

  // Verify pipeline terminates with the same `iter_args` types as the pipeline.
  auto iterArgs = op.iter_args();
  TypeRange terminatorArgTypes = iterArgs.getTypes();
  TypeRange pipelineArgTypes = pipeline.iterArgs().getTypes();
  if (terminatorArgTypes != pipelineArgTypes)
    return op.emitOpError("'iter_args' types (")
           << terminatorArgTypes << ") must match pipeline 'iter_args' types ("
           << pipelineArgTypes << ")";

  // Verify `iter_args` are defined by a pipeline stage.
  for (auto iterArg : iterArgs)
    if (iterArg.getDefiningOp<PipelineStageOp>() == nullptr)
      return op.emitOpError(
          "'iter_args' must be defined by a 'staticlogic.pipeline.stage'");

  // Verify pipeline terminates with the same result types as the pipeline.
  auto results = op.results();
  TypeRange terminatorResultTypes = results.getTypes();
  TypeRange pipelineResultTypes = pipeline.getResultTypes();
  if (terminatorResultTypes != pipelineResultTypes)
    return op.emitOpError("'results' types (")
           << terminatorResultTypes << ") must match pipeline result types ("
           << pipelineResultTypes << ")";

  // Verify `results` are defined by a pipeline stage.
  for (auto result : results)
    if (result.getDefiningOp<PipelineStageOp>() == nullptr)
      return op.emitOpError(
          "'results' must be defined by a 'staticlogic.pipeline.stage'");

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogic.cpp.inc"

void StaticLogicDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/StaticLogic/StaticLogic.cpp.inc"
      >();
}
