//===- FIRRTL/IR/Dialect.h - FIRRTL dialect declaration ---------*- C++ -*-===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_IR_DIALECT_H
#define SPT_DIALECT_FIRRTL_IR_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace spt {
namespace firrtl {
using namespace mlir;
class FIRRTLType;

class FIRRTLDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  explicit FIRRTLDialect(MLIRContext *context);
  ~FIRRTLDialect();

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type, DialectAsmPrinter &) const override;

  static StringRef getDialectNamespace() { return "firrtl"; }
};

/// If the specified attribute list has a firrtl.name attribute, return its
/// value.
StringAttr getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs);

} // namespace firrtl
} // namespace spt

// Pull in all enum type definitions and utility function declarations.
#include "spt/Dialect/FIRRTL/FIRRTLEnums.h.inc"

#endif // SPT_DIALECT_FIRRTL_IR_DIALECT_H
