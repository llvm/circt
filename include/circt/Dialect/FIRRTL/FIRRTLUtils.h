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
namespace hw {
struct InnerSymbolNamespace;
} // namespace hw

namespace firrtl {
/// Emit a connect between two values.
void emitConnect(OpBuilder &builder, Location loc, Value lhs, Value rhs);
void emitConnect(ImplicitLocOpBuilder &builder, Value lhs, Value rhs);

/// Utiility for generating a constant attribute.
IntegerAttr getIntAttr(Type type, const APInt &value);

/// Utility for generating a constant zero attribute.
IntegerAttr getIntZerosAttr(Type type);

/// Utility for generating a constant all ones attribute.
IntegerAttr getIntOnesAttr(Type type);

/// Return the single assignment to a Property value.
PropAssignOp getPropertyAssignment(FIRRTLPropertyValue value);

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
bool walkDrivers(FIRRTLBaseValue value, bool lookThroughWires,
                 bool lookThroughNodes, bool lookThroughCasts,
                 WalkDriverCallback callback);

/// Get the FieldRef from a value.  This will travel backwards to through the
/// IR, following Subfield and Subindex to find the op which declares the
/// location.  Optionally look through recognized cast operations, which
/// likely will result in source having slightly different type.
FieldRef getFieldRefFromValue(Value value, bool lookThroughCasts = false);

/// Get the delta indexing from a value, as a FieldRef.
FieldRef getDeltaRef(Value value, bool lookThroughCasts = false);

/// Get a string identifier representing the FieldRef.  Return this string and a
/// boolean indicating if a valid "root" for the identifier was found.  If
/// nameSafe is true, this will generate a string that is better suited for
/// naming something in the IR.  E.g., if the fieldRef is a subfield of a
/// subindex, without name safe the output would be:
///
///   foo[42].bar
///
/// With nameSafe, this would be:
///
///   foo_42_bar
std::pair<std::string, bool> getFieldName(const FieldRef &fieldRef,
                                          bool nameSafe = false);

Value getValueByFieldID(ImplicitLocOpBuilder builder, Value value,
                        unsigned fieldID);

/// Walk leaf ground types in the `firrtlType` and apply the function `fn`.
/// The first argument of `fn` is field ID, and the second argument is a
/// leaf ground type, and the third argument indicates if the element was
/// flipped in a bundle.
void walkGroundTypes(
    FIRRTLType firrtlType,
    llvm::function_ref<void(uint64_t, FIRRTLBaseType, bool)> fn);

//===----------------------------------------------------------------------===//
// Inner symbol and InnerRef helpers.
//===----------------------------------------------------------------------===//

/// Return the inner sym target for the specified value and fieldID.
/// If root is a blockargument, this must be FModuleLike.
hw::InnerSymTarget getTargetFor(FieldRef ref);

/// Get FieldRef pointing to the specified inner symbol target, which must be
/// valid. Returns null FieldRef if target points to something with no value,
/// such as a port of an external module.
FieldRef getFieldRefForTarget(const hw::InnerSymTarget &ist);

/// Ensure that the the InnerSymAttr has a symbol on the field specified.
/// Returns the updated InnerSymAttr as well as the name of the symbol attached
/// to the specified field.
std::pair<hw::InnerSymAttr, StringAttr>
getOrAddInnerSym(MLIRContext *context, hw::InnerSymAttr attr, uint64_t fieldID,
                 llvm::function_ref<hw::InnerSymbolNamespace &()> getNamespace);

/// Returns an inner symbol identifier for the specified target (op or port),
/// adding one if necessary.
StringAttr
getOrAddInnerSym(const hw::InnerSymTarget &target,
                 llvm::function_ref<hw::InnerSymbolNamespace &()> getNamespace);

using GetNamespaceCallback =
    llvm::function_ref<hw::InnerSymbolNamespace &(FModuleLike mod)>;

/// Returns an inner symbol identifier for the specified target (op or port),
/// adding one if necessary.
StringAttr getOrAddInnerSym(const hw::InnerSymTarget &target,
                            GetNamespaceCallback getNamespace);

/// Obtain an inner reference to the target (operation or port),
/// adding an inner symbol as necessary.
hw::InnerRefAttr getInnerRefTo(const hw::InnerSymTarget &target,
                               GetNamespaceCallback getNamespace);

/// Returns an inner symbol identifier for the specified operation, adding one
/// if necessary.
static inline StringAttr getOrAddInnerSym(Operation *op,
                                          GetNamespaceCallback getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(op), getNamespace);
}
/// Returns an inner symbol identifier for the specified operation's field
/// adding one if necessary.
static inline StringAttr getOrAddInnerSym(Operation *op, uint64_t fieldID,
                                          GetNamespaceCallback getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(op, fieldID), getNamespace);
}

/// Obtain an inner reference to an operation, possibly adding an inner symbol.
static inline hw::InnerRefAttr
getInnerRefTo(Operation *op, GetNamespaceCallback getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(op), getNamespace);
}

/// Obtain an inner reference to an operation's field, possibly adding an inner
/// symbol.
static inline hw::InnerRefAttr
getInnerRefTo(Operation *op, uint64_t fieldID,
              GetNamespaceCallback getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(op, fieldID), getNamespace);
}

/// Returns an inner symbol identifier for the specified port, adding one if
/// necessary.
static inline StringAttr getOrAddInnerSym(FModuleLike mod, size_t portIdx,
                                          GetNamespaceCallback getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(portIdx, mod), getNamespace);
}

/// Returns an inner symbol identifier for the specified port's field, adding
/// one if necessary.
static inline StringAttr getOrAddInnerSym(FModuleLike mod, size_t portIdx,
                                          uint64_t fieldID,
                                          GetNamespaceCallback getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(portIdx, mod, fieldID),
                          getNamespace);
}

/// Obtain an inner reference to a port, possibly adding an inner symbol.
static inline hw::InnerRefAttr
getInnerRefTo(FModuleLike mod, size_t portIdx,
              GetNamespaceCallback getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(portIdx, mod), getNamespace);
}

/// Obtain an inner reference to a port's field, possibly adding an inner
/// symbol.
static inline hw::InnerRefAttr
getInnerRefTo(FModuleLike mod, size_t portIdx, uint64_t fieldID,
              GetNamespaceCallback getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(portIdx, mod, fieldID), getNamespace);
}

//===----------------------------------------------------------------------===//
// Type utilities
//===----------------------------------------------------------------------===//

/// If it is a base type, return it as is. If reftype, return wrapped base type.
/// Otherwise, return null.
inline FIRRTLBaseType getBaseType(Type type) {
  return TypeSwitch<Type, FIRRTLBaseType>(type)
      .Case<FIRRTLBaseType>([](auto base) { return base; })
      .Case<LHSType>([](auto lhs) { return lhs.getType(); })
      .Case<RefType>([](auto ref) { return ref.getType(); })
      .Default([](Type type) { return nullptr; });
}

/// Get base type if isa<> the requested type, else null.
template <typename T>
inline T getBaseOfType(Type type) {
  return dyn_cast_or_null<T>(getBaseType(type));
}

/// Return a FIRRTLType with its base type component mutated by the given
/// function. (i.e., ref<T> -> ref<f(T)> and T -> f(T)).
inline FIRRTLType mapBaseType(FIRRTLType type,
                              function_ref<FIRRTLBaseType(FIRRTLBaseType)> fn) {
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<FIRRTLBaseType>([&](auto base) { return fn(base); })
      .Case<RefType>([&](auto ref) {
        return RefType::get(fn(ref.getType()), ref.getForceable(),
                            ref.getLayer());
      });
}

/// Return a FIRRTLType with its base type component mutated by the given
/// function. Return null when the function returns null.
/// (i.e., ref<T> -> ref<f(T)> if f(T) != null else null, and T -> f(T)).
inline FIRRTLType
mapBaseTypeNullable(FIRRTLType type,
                    function_ref<FIRRTLBaseType(FIRRTLBaseType)> fn) {
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<FIRRTLBaseType>([&](auto base) { return fn(base); })
      .Case<RefType>([&](auto ref) -> FIRRTLType {
        auto result = fn(ref.getType());
        if (!result)
          return {};
        return RefType::get(result, ref.getForceable(), ref.getLayer());
      });
}

/// Given a type, return the corresponding lowered type for the HW dialect.
/// Non-FIRRTL types are simply passed through. This returns a null type if it
/// cannot be lowered. The optional function is required to specify how to lower
/// AliasTypes.
Type lowerType(
    Type type, std::optional<Location> loc = {},
    llvm::function_ref<hw::TypeAliasType(Type, BaseTypeAliasType, Location)>
        getTypeDeclFn = {});

//===----------------------------------------------------------------------===//
// Parser-related utilities
//
// These cannot always be relegated to the parser and sometimes need to be
// available for passes.  This has specifically come up for Annotation lowering
// where there is FIRRTL stuff that needs to be parsed out of an annotation.
//===----------------------------------------------------------------------===//

/// Parse a string that may encode a FIRRTL location into a LocationAttr.
std::pair<bool, std::optional<mlir::LocationAttr>> maybeStringToLocation(
    StringRef spelling, bool skipParsing, StringAttr &locatorFilenameCache,
    FileLineColLoc &fileLineColLocCache, MLIRContext *context);

// Parse a format string and build operations for FIRRTL "special"
// substitutions.
//
// This function handles:
// - percent format strings (%b, %d, %x, %c, %%)
// - special format strings ({{SimulationTime}}, {{HierarchicalModuleName}})
//
// The formatStringResult output parameter is set to the validated format
// string. The operands output parameter is set to the list of actual operands
// (including special ops created for {{...}} substitutions).
mlir::ParseResult
parseFormatString(mlir::OpBuilder &builder, mlir::Location loc,
                  llvm::StringRef formatString,
                  llvm::ArrayRef<mlir::Value> specOperands,
                  mlir::StringAttr &formatStringResult,
                  llvm::SmallVectorImpl<mlir::Value> &operands);

//===----------------------------------------------------------------------===//
// File utilities
//===----------------------------------------------------------------------===//

/// Truncate `a` to the common prefix of `a` and `b`.
void makeCommonPrefix(SmallString<64> &a, StringRef b);

//===----------------------------------------------------------------------===//
// Object related utilities
//===----------------------------------------------------------------------===//

/// Add the tracker annotation to the op and get a PathOp to the op.
PathOp createPathRef(Operation *op, hw::HierPathOp nla,
                     mlir::ImplicitLocOpBuilder &builderOM);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
