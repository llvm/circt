//===- ESIOps.cpp - ESI op code defs ----------------------------*- C++ -*-===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/SV/Ops.h"
#include "circt/Dialect/SV/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ChannelBuffer functions.
//===----------------------------------------------------------------------===//

static ParseResult parseChannelBuffer(OpAsmParser &parser,
                                      OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3,
                              /*delimiter=*/OpAsmParser::Delimiter::None))
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

  auto i1 = IntegerType::get(1, result.getContext());
  if (parser.resolveOperands(operands, {i1, i1, outputType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, ChannelBuffer &op) {
  p << "esi.buffer " << op.clk() << ", " << op.rstn() << ", " << op.input()
    << " ";
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

  SmallVector<OpAsmParser::OperandType, 4> operands;
  Type innerOutputType;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();
  auto type =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({type});

  auto i1 = IntegerType::get(1, result.getContext());
  if (parser.resolveOperands(operands, {i1, i1, type}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, PipelineStage &op) {
  p << "esi.stage " << op.clk() << ", " << op.rstn() << ", " << op.input()
    << " ";
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

/// If 'iface' looks like an ESI interface, return the inner data type.
static Type getEsiDataType(circt::sv::InterfaceOp iface) {
  using namespace circt::sv;
  if (!iface.lookupSymbol<InterfaceSignalOp>("valid"))
    return Type();
  if (!iface.lookupSymbol<InterfaceSignalOp>("ready"))
    return Type();
  auto dataSig = iface.lookupSymbol<InterfaceSignalOp>("data");
  if (!dataSig)
    return Type();
  return dataSig.type();
}

/// Verify that the modport type of 'modportArg' points to an interface which
/// looks like an ESI interface and the inner data from said interface matches
/// the chan type's inner data type.
static LogicalResult verifySVInterface(Operation *op,
                                       circt::sv::ModportType modportType,
                                       ChannelPort chanType) {
  auto modport =
      SymbolTable::lookupNearestSymbolFrom<circt::sv::InterfaceModportOp>(
          op, modportType.getModport());
  if (!modport)
    return op->emitError("Could not find modport ")
           << modportType.getModport() << " in symbol table.";
  auto iface = cast<circt::sv::InterfaceOp>(modport.getParentOp());
  Type esiDataType = getEsiDataType(iface);
  if (!esiDataType)
    return op->emitOpError("Interface is not a valid ESI interface.");
  if (esiDataType != chanType.getInner())
    return op->emitOpError("Operation specifies ")
           << chanType << " but type inside doesn't match interface data type "
           << esiDataType << ".";
  return success();
}

static LogicalResult verifyWrapSVInterface(WrapSVInterface &op) {
  auto modportType = op.iface().getType().cast<circt::sv::ModportType>();
  auto chanType = op.output().getType().cast<ChannelPort>();
  return verifySVInterface(op, modportType, chanType);
}

static LogicalResult verifyUnwrapSVInterface(UnwrapSVInterface &op) {
  auto modportType = op.outIface().getType().cast<circt::sv::ModportType>();
  auto chanType = op.chanInput().getType().cast<ChannelPort>();
  return verifySVInterface(op, modportType, chanType);
}

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"
