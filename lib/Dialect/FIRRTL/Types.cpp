//===- Types.cpp - Implement the FIRRTL dialect type system ---------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"

using namespace spt;
using namespace firrtl;

/// Parse a type registered to this dialect.
Type FIRRTLDialect::parseType(DialectAsmParser &parser) const {
  StringRef tyData = parser.getFullSymbolSpec();

  if (tyData == "sint")
    return UIntType::get(getContext());
  if (tyData == "uint")
    return UIntType::get(getContext());
  if (tyData == "clock")
    return ClockType::get(getContext());
  if (tyData == "reset")
    return ResetType::get(getContext());
  if (tyData == "analog")
    return AnalogType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown firrtl type");
  return Type();
}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  default:
    assert(0 && "unknown dialect type to print");
  case FIRRTLTypes::SInt:
    os.getStream() << "sint";
    break;
  case FIRRTLTypes::UInt:
    os.getStream() << "uint";
    break;
  case FIRRTLTypes::Clock:
    os.getStream() << "clock";
    break;
  case FIRRTLTypes::Reset:
    os.getStream() << "reset";
    break;

  // Derived types.
  case FIRRTLTypes::Analog:
    os.getStream() << "analog";
    break;
  }
}
