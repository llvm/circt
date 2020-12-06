//===- ESIOps.cpp - ESI op code defs ----------------------------*- C++ -*-===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ChannelBuffer functions.
//===----------------------------------------------------------------------===//

static ParseResult parseChannelBuffer(OpAsmParser &parser,
                                      OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  OpAsmParser::OperandType inputOperand;
  if (parser.parseOperand(inputOperand))
    return failure();

  ChannelBufferOptions optionsAttr;
  if (parser.parseAttribute(optionsAttr,
                            parser.getBuilder().getType<NoneType>(), "options",
                            result.attributes))
    return failure();

  Type innerOutputType;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();
  auto outputType =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({outputType});

  if (parser.resolveOperands({inputOperand}, {outputType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, ChannelBuffer &op) {
  p << "esi.buffer " << op.input() << " ";
  p.printAttributeWithoutType(op.options());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"options"});
  p << " : " << op.output().getType().cast<ChannelPort>().getInner();
}

//===----------------------------------------------------------------------===//
// PipelineStage functions.
//===----------------------------------------------------------------------===//

static ParseResult parsePipelineStage(OpAsmParser &parser,
                                      OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  OpAsmParser::OperandType inputOperand;
  Type innerOutputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();
  auto type =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({type});

  if (parser.resolveOperands({inputOperand}, {type}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, PipelineStage &op) {
  p << "esi.stage " << op.input() << " ";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.output().getType().cast<ChannelPort>().getInner();
}

//===----------------------------------------------------------------------===//
// Wrap / unwrap.
//===----------------------------------------------------------------------===//

static ParseResult parseWrapValidReady(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::OperandType, 2> opList;
  Type innerOutputType;
  if (parser.parseOperandList(opList, 2, OpAsmParser::Delimiter::None) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();

  auto boolType = parser.getBuilder().getI1Type();
  auto outputType =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({outputType, boolType});
  if (parser.resolveOperands(opList, {innerOutputType, boolType},
                             inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void print(OpAsmPrinter &p, WrapValidReady &op) {
  p << "esi.wrap.vr " << op.data() << ", " << op.valid();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.output().getType().cast<ChannelPort>().getInner();
}

static ParseResult parseUnwrapValidReady(OpAsmParser &parser,
                                         OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::OperandType, 2> opList;
  Type outputType;
  if (parser.parseOperandList(opList, 2, OpAsmParser::Delimiter::None) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(outputType))
    return failure();

  auto inputType =
      ChannelPort::get(parser.getBuilder().getContext(), outputType);

  auto boolType = parser.getBuilder().getI1Type();

  result.addTypes({inputType.getInner(), boolType});
  if (parser.resolveOperands(opList, {inputType, boolType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, UnwrapValidReady &op) {
  p << "esi.unwrap.vr " << op.input() << ", " << op.ready();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.output().getType();
}

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"
