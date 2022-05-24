//===- FIRRTLUtils.h - FIRRTL IR Utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utilties to help generate and process FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace circt {
namespace firrtl {
/// Emit a connect between two values.
void emitConnect(OpBuilder &builder, Location loc, Value lhs, Value rhs);
void emitConnect(ImplicitLocOpBuilder &builder, Value lhs, Value rhs);

/// Utiility for generating a constant attribute.
IntegerAttr getIntAttr(Type type, const APInt &value);

/// Utility for generating a constant zero attribute.
IntegerAttr getIntZerosAttr(Type type);

/// Return the module-scoped driver of a value
Value getModuleScopedDriver(Value val, bool lookThroughWires,
                            bool lookThroughNodes, bool lookThroughCasts);

/// Return true if a value is module-scoped driven by a value of a specific
/// type.
template <typename A, typename... B>
static bool isModuleScopedDrivenBy(Value val, bool lookThroughWires,
                                   bool lookThroughNodes,
                                   bool lookThroughCasts) {
  val = getModuleScopedDriver(val, lookThroughWires, lookThroughNodes,
                              lookThroughCasts);

  if (!val)
    return false;

  auto *op = val.getDefiningOp();
  if (!op)
    return false;

  return isa<A, B...>(op);
}

Value getValueByFieldID(ImplicitLocOpBuilder builder, Value value,
                        unsigned fieldID);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
