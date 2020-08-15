//===- SV/IR/Dialect.h - SV dialect declaration -----------------*- C++ -*-===//
//
// This file defines an SV MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_DIALECT_H
#define CIRCT_DIALECT_SV_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace circt {
namespace sv {
using namespace mlir;

class SVDialect : public Dialect {
public:
  explicit SVDialect(MLIRContext *context);
  ~SVDialect();

  static StringRef getDialectNamespace() { return "sv"; }
};

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_DIALECT_H
