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
// Structure operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InterfaceOp

static ParseResult parseInterfaceOp(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None))
    return failure();

  InterfaceOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  return success();
}

static void printInterfaceOp(OpAsmPrinter &p, InterfaceOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.getName());
  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// InterfaceSignalOp

static ParseResult parseInterfaceSignalOp(OpAsmParser &parser,
                                          OperationState &result) {
  StringAttr nameAttr;
  TypeAttr type;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(type, "type", result.attributes))
    return failure();

  return success();
}

static void printInterfaceSignalOp(OpAsmPrinter &p, InterfaceSignalOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.getName());
  p << " : ";
  p.printAttribute(op.getAttr("type"));
}

//===----------------------------------------------------------------------===//
// InterfaceModPortOp

static ParseResult parseInterfaceModPortOp(OpAsmParser &parser,
                                           OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseLParen())
    return failure();

  auto context = result.getContext();
  SmallVector<Attribute, 8> ports;
  while (true) {
    NamedAttrList tmpAttrs;
    StringAttr directionAttr;
    StringAttr signalAttr;
    if (parser.parseSymbolName(signalAttr, "signal", tmpAttrs) ||
        parser.parseColon() ||
        parser.parseAttribute(directionAttr, "direction", tmpAttrs))
      break;

    auto directionEnum =
        symbolizeModPortDirectionAttr(directionAttr.getValue());

    if (!directionEnum.hasValue()) {
      return failure();
    }

    auto signalRef = FlatSymbolRefAttr::get(signalAttr.getValue(), context);

    ports.push_back(ModPortStructAttr::get(directionAttr, signalRef, context));

    if (!parser.parseOptionalRParen())
      break;

    parser.parseOptionalComma();
  }

  result.addAttribute("ports", ArrayAttr::get(ports, context));

  return success();
}

static void printInterfaceModPortOp(OpAsmPrinter &p, InterfaceModPortOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.getName());
  p << " (";

  auto ports = op.ports();
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    auto port = ports[i].dyn_cast<ModPortStructAttr>();

    p.printSymbolName(port.signal().getRootReference());
    p << " : ";
    p << port.direction();

    if (i < e - 1) {
      p << ", ";
    }
  }

  p << ')';
}

static LogicalResult verifyInterfaceModPortOp(InterfaceModPortOp op) {
  for (auto port : op.ports()) {
    if (!port.dyn_cast<ModPortStructAttr>()) {
      op.emitOpError("should have an array of ModPortStructAttr attributes");
      return failure();
    }
  }

  return success();
}

// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.cpp.inc"
#include "circt/Dialect/SV/SVEnums.cpp.inc"

namespace circt {
#include "circt/Dialect/SV/SVStructs.cpp.inc"
} // namespace circt
