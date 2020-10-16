//===- RTL/IR/Dialect.h - RTL dialect declaration ---------------*- C++ -*-===//
//
// This file defines an RTL MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_DIALECT_H
#define CIRCT_DIALECT_RTL_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace circt {
namespace rtl {
using namespace mlir;

class RTLDialect : public Dialect {
public:
  explicit RTLDialect(MLIRContext *context);
  ~RTLDialect();

  static StringRef getDialectNamespace() { return "rtl"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

} // namespace rtl
} // namespace circt

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/RTL/RTLEnums.h.inc"

#endif // CIRCT_DIALECT_RTL_DIALECT_H
