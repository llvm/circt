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
#include "circt/Dialect/FIRRTL/Namespace.h"
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

/// Return the module-scoped driver of a value only looking through one connect.
Value getDriverFromConnect(Value val);

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

/// Get the FieldRef from a value.  This will travel backwards to through the
/// IR, following Subfield and Subindex to find the op which declares the
/// location.
FieldRef getFieldRefFromValue(Value value);

/// Get a string identifier representing the FieldRef.
std::string getFieldName(const FieldRef &fieldRef);
std::string getFieldName(const FieldRef &fieldRef, bool &rootKnown);

Value getValueByFieldID(ImplicitLocOpBuilder builder, Value value,
                        unsigned fieldID);

/// Returns an operation's `inner_sym`, adding one if necessary.
StringAttr
getOrAddInnerSym(Operation *op, StringRef nameHint, FModuleOp mod,
                 std::function<ModuleNamespace &(FModuleOp)> getNamespace);

/// Obtain an inner reference to an operation, possibly adding an `inner_sym`
/// to that operation.
hw::InnerRefAttr
getInnerRefTo(Operation *op, StringRef nameHint,
              std::function<ModuleNamespace &(FModuleOp)> getNamespace);

/// Returns a port's `inner_sym`, adding one if necessary.
StringAttr
getOrAddInnerSym(FModuleLike mod, size_t portIdx, StringRef nameHint,
                 std::function<ModuleNamespace &(FModuleLike)> getNamespace);

/// Obtain an inner reference to a port, possibly adding an `inner_sym`
/// to the port.
hw::InnerRefAttr
getInnerRefTo(FModuleLike mod, size_t portIdx, StringRef nameHint,
              std::function<ModuleNamespace &(FModuleLike)> getNamespace);
} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
