//===- Dialect.cpp - Implement the FIRRTL dialect -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace spt;
using namespace firrtl;

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


#define GET_OP_CLASSES
#include "spt/Dialect/FIRRTL/IR/FIRRTL.cpp.inc"
