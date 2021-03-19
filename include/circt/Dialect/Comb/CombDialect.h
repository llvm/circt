//===- CombDialect.h - Comb dialect declaration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Combinational MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COMB_COMBDIALECT_H
#define CIRCT_DIALECT_COMB_COMBDIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace comb {
using namespace mlir;

class CombDialect : public Dialect {
public:
  explicit CombDialect(MLIRContext *context);
  ~CombDialect();

  static StringRef getDialectNamespace() { return "comb"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

} // namespace comb
} // namespace circt

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/Comb/CombEnums.h.inc"

#endif // CIRCT_DIALECT_COMB_COMBDIALECT_H
