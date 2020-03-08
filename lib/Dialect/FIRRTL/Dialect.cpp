//===- Dialect.cpp - Implement the FIRRTL dialect -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace spt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

FIRRTLDialect::FIRRTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<UIntType>();
  addOperations<
#define GET_OP_LIST
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
      >();
      
  // Support unknown operations because not all LLVM operations are registered.
  // FIXME(clattner): remove this.
  allowUnknownOperations();
}

FIRRTLDialect::~FIRRTLDialect() {
}

//===----------------------------------------------------------------------===//
// Type specifications.
//===----------------------------------------------------------------------===//

/// Parse a type registered to this dialect.
Type FIRRTLDialect::parseType(DialectAsmParser &parser) const {
  StringRef tyData = parser.getFullSymbolSpec();
  
  if (tyData == "uint")
    return UIntType::get(getContext());
  
  parser.emitError(parser.getNameLoc(), "unknown firrtl type");
  return Type();
}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  auto uintType = type.dyn_cast<UIntType>();
  assert(uintType && "printing wrong type");
  os.getStream() << "uint";
}

//===----------------------------------------------------------------------===//
// Module and Circuit Ops.
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, CircuitOp op) {
  p << op.getOperationName() << " ";
  p.printAttribute(op.nameAttr());

  p.printOptionalAttrDictWithKeyword(op.getAttrs(), {"name"});

  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseCircuitOp(OpAsmParser &parser, OperationState &result) {
  // Parse the module name.
  StringAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*regionArgs*/{}, /*argTypes*/{}))
    return failure();

  CircuitOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  return success();
}

static void print(OpAsmPrinter &p, FModuleOp op) {
  p << op.getOperationName() << " ";
  p.printAttribute(op.nameAttr());

  p.printOptionalAttrDictWithKeyword(op.getAttrs(), {"name"});

  p.printRegion(op.body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseFModuleOp(OpAsmParser &parser, OperationState &result) {
  // Parse the module name.
  StringAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*regionArgs*/{}, /*argTypes*/{}))
    return failure();

  FModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  return success();
}
#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
