//===- FIRRTLDialect.h - FIRRTL dialect declaration ------------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_DIALECT_H
#define CIRCT_DIALECT_FIRRTL_DIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace firrtl {
class FIRRTLType;

class FIRRTLDialect : public mlir::Dialect {
public:
  /// Create the dialect in the given `context`.
  explicit FIRRTLDialect(mlir::MLIRContext *context);
  ~FIRRTLDialect();

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type, mlir::DialectAsmPrinter &) const override;

  mlir::Operation *materializeConstant(mlir::OpBuilder &builder,
                                       mlir::Attribute value, mlir::Type type,
                                       mlir::Location loc) override;

  static llvm::StringRef getDialectNamespace() { return "firrtl"; }
};

/// If the specified attribute list has a firrtl.name attribute, return its
/// value.
mlir::StringAttr getFIRRTLNameAttr(llvm::ArrayRef<mlir::NamedAttribute> attrs);

} // namespace firrtl
} // namespace circt

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/FIRRTL/FIRRTLEnums.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_IR_DIALECT_H
