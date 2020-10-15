//===- Ops.cpp - Implement the SV operations ------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace circt;
using namespace sv;

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IfDefOp

void IfDefOp::build(OpBuilder &odsBuilder, OperationState &result,
                    StringRef cond) {
  result.addAttribute("cond", odsBuilder.getStringAttr(cond));
  Region *body = result.addRegion();
  IfDefOp::ensureTerminator(*body, odsBuilder, result.location);
}

static ParseResult parseIfDefOp(OpAsmParser &parser, OperationState &result) {
  StringAttr cond;
  Region *body = result.addRegion();
  if (parser.parseAttribute(cond, "cond", result.attributes) ||
      parser.parseRegion(*body, llvm::None, llvm::None) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IfDefOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

static void printIfDefOp(OpAsmPrinter &p, IfDefOp op) {
  p << op.getOperationName() << ' ';
  p.printAttribute(op.condAttr());
  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(op.getAttrs(), {"cond"});
}

//===----------------------------------------------------------------------===//
// IfOp

void IfOp::build(OpBuilder &odsBuilder, OperationState &result, Value cond) {
  result.addOperands(cond);
  Region *body = result.addRegion();
  IfOp::ensureTerminator(*body, odsBuilder, result.location);
}

static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType cond;
  Region *body = result.addRegion();
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, parser.getBuilder().getI1Type(),
                            result.operands) ||
      parser.parseRegion(*body, llvm::None, llvm::None) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  IfOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

static void printIfOp(OpAsmPrinter &p, IfOp op) {
  p << op.getOperationName() << ' ' << op.cond();
  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// AlwaysAtPosEdgeOp

void AlwaysAtPosEdgeOp::build(OpBuilder &odsBuilder, OperationState &result,
                              Value clock) {
  result.addOperands(clock);
  Region *body = result.addRegion();
  AlwaysAtPosEdgeOp::ensureTerminator(*body, odsBuilder, result.location);
}

static ParseResult parseAlwaysAtPosEdgeOp(OpAsmParser &parser,
                                          OperationState &result) {
  OpAsmParser::OperandType clock;
  Region *body = result.addRegion();
  if (parser.parseOperand(clock) ||
      parser.resolveOperand(clock, parser.getBuilder().getI1Type(),
                            result.operands) ||
      parser.parseRegion(*body, llvm::None, llvm::None) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  AlwaysAtPosEdgeOp::ensureTerminator(*body, parser.getBuilder(),
                                      result.location);
  return success();
}

static void printAlwaysAtPosEdgeOp(OpAsmPrinter &p, AlwaysAtPosEdgeOp op) {
  p << op.getOperationName() << ' ' << op.clock();
  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// TypeDecl operations
//===----------------------------------------------------------------------===//

static ParseResult parseModportStructs(OpAsmParser &parser,
                                       ArrayAttr &portsAttr) {
  if (parser.parseLParen())
    return failure();

  auto context = parser.getBuilder().getContext();

  SmallVector<Attribute, 8> ports;
  do {
    NamedAttrList tmpAttrs;
    StringAttr direction;
    FlatSymbolRefAttr signal;
    if (parser.parseAttribute(direction, "port", tmpAttrs) ||
        parser.parseAttribute(signal, "signal", tmpAttrs))
      return failure();

    ports.push_back(ModportStructAttr::get(direction, signal, context));
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();

  portsAttr = ArrayAttr::get(ports, context);

  return success();
}

static void printModportStructs(OpAsmPrinter &p, ArrayAttr portsAttr) {
  p << " (";
  for (size_t i = 0, e = portsAttr.size(); i != e; ++i) {
    auto port = portsAttr[i].dyn_cast<ModportStructAttr>();
    p << port.direction();
    p << ' ';
    p.printSymbolName(port.signal().getRootReference());
    if (i < e - 1) {
      p << ", ";
    }
  }
  p << ')';
}

// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.cpp.inc"
#include "circt/Dialect/SV/SVEnums.cpp.inc"
#include "circt/Dialect/SV/SVStructs.cpp.inc"
