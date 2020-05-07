//===- RTL/IR/Dialect.h - RTL dialect declaration ---------------*- C++ -*-===//
//
// This file defines an RTL MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRT_DIALECT_RTL_DIALECT_H
#define CIRT_DIALECT_RTL_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace cirt {
namespace rtl {
using namespace mlir;

class RTLDialect : public Dialect {
public:
  explicit RTLDialect(MLIRContext *context);
  ~RTLDialect();

  static StringRef getDialectNamespace() { return "rtl"; }
};

} // namespace rtl
} // namespace cirt

#endif // CIRT_DIALECT_RTL_DIALECT_H
