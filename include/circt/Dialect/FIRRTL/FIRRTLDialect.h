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

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

namespace circt {
namespace firrtl {

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

private:
  /// Register all FIRRTL types.
  void registerTypes();
};

/// If the specified attribute list has a firrtl.name attribute, return its
/// value.
StringAttr getFIRRTLNameAttr(ArrayRef<NamedAttribute> attrs);

} // namespace firrtl
} // namespace circt

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/FIRRTL/FIRRTLEnums.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_IR_DIALECT_H
