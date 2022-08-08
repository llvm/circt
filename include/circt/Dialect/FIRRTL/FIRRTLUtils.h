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

/// Return the value that drives another FIRRTL value within module scope.  This
/// is parameterized by looking through or not through certain constructs.
Value getValueSource(Value val, bool lookThroughWires, bool lookThroughNodes,
                     bool lookThroughCasts);

/// Return the value that drives another FIRRTL value within module scope.  This
/// is parameterized by looking through or not through certain constructs.  This
/// assumes a single driver and should only be run after `ExpandWhens`.
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

/// Walk all the drivers of a value, passing in the connect operations drive the
/// value. If the value is an aggregate it will find connects to subfields. If
/// the callback returns false, this function will stop walking.  Returns false
/// if walking was broken, and true otherwise.
using WalkDriverCallback =
    llvm::function_ref<bool(const FieldRef &dst, const FieldRef &src)>;
bool walkDrivers(Value value, bool lookThroughWires,
                 bool lookTWalkDriverCallbackhroughNodes, bool lookThroughCasts,
                 WalkDriverCallback callback);

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

//===----------------------------------------------------------------------===//
// Parser-related utilities
//
// These cannot always be relegated to the parser and sometimes need to be
// available for passes.  This has specifically come up for Annotation lowering
// where there is FIRRTL stuff that needs to be parsed out of an annotation.
//===----------------------------------------------------------------------===//

/// Parse a string that may encode a FIRRTL location into a LocationAttr.
std::pair<bool, Optional<mlir::LocationAttr>> maybeStringToLocation(
    StringRef spelling, bool skipParsing, StringAttr &locatorFilenameCache,
    FileLineColLoc &fileLineColLocCache, MLIRContext *context);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
