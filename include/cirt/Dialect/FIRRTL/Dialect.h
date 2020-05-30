//===- FIRRTL/IR/Dialect.h - FIRRTL dialect declaration ---------*- C++ -*-===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRT_DIALECT_FIRRTL_DIALECT_H
#define CIRT_DIALECT_FIRRTL_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace cirt {
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

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  static StringRef getDialectNamespace() { return "firrtl"; }
};

/// If the specified attribute list has a firrtl.name attribute, return its
/// value.
StringAttr getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs);

/// Register all of the FIRRTL transformation passes with the PassManager.
void registerFIRRTLPasses();

} // namespace firrtl
} // namespace cirt

// Pull in all enum type definitions and utility function declarations.
#include "cirt/Dialect/FIRRTL/FIRRTLEnums.h.inc"

#endif // CIRT_DIALECT_FIRRTL_IR_DIALECT_H
