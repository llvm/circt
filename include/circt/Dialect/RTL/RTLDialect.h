//===- RTLDialect.h - RTL dialect declaration -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an RTL MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_RTLDIALECT_H
#define CIRCT_DIALECT_RTL_RTLDIALECT_H

#include "mlir/IR/Dialect.h"

namespace circt {
namespace rtl {

class RTLDialect : public mlir::Dialect {
public:
  explicit RTLDialect(mlir::MLIRContext *context);
  ~RTLDialect();

  static llvm::StringRef getDialectNamespace() { return "rtl"; }

  mlir::Operation *materializeConstant(mlir::OpBuilder &builder,
                                       mlir::Attribute value, mlir::Type type,
                                       mlir::Location loc) override;

  /// Parses a type registered to this dialect
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

} // namespace rtl
} // namespace circt

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/RTL/RTLEnums.h.inc"

#endif // CIRCT_DIALECT_RTL_RTLDIALECT_H
