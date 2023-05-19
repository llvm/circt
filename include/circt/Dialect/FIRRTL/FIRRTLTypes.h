//===- FIRRTLTypes.h - FIRRTL Type System -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type system for the FIRRTL Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_TYPES_H
#define CIRCT_DIALECT_FIRRTL_TYPES_H

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypeInterfaces.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace firrtl {
namespace detail {
struct FIRRTLBaseTypeStorage;
struct WidthTypeStorage;
struct BundleTypeStorage;
struct VectorTypeStorage;
struct FEnumTypeStorage;
struct CMemoryTypeStorage;
struct RefTypeStorage;
} // namespace detail.

class ClockType;
class ResetType;
class AsyncResetType;
class SIntType;
class UIntType;
class AnalogType;
class BundleType;
class OpenBundleType;
class OpenVectorType;
class FVectorType;
class FEnumType;
class RefType;
class StringType;

/// A collection of bits indicating the recursive properties of a type.
struct RecursiveTypeProperties {
  /// Whether the type only contains passive elements.
  bool isPassive : 1;
  /// Whether the type contains a reference type.
  bool containsReference : 1;
  /// Whether the type contains an analog type.
  bool containsAnalog : 1;
  /// Whether the type contains a const type.
  bool containsConst : 1;
  /// Whether the type has any uninferred bit widths.
  bool hasUninferredWidth : 1;
  /// Whether the type has any uninferred reset.
  bool hasUninferredReset : 1;
};

// This is a common base class for all FIRRTL types.
class FIRRTLType : public Type {
public:
  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return llvm::isa<FIRRTLDialect>(type.getDialect());
  }

  /// Return the recursive properties of the type, containing the `isPassive`,
  /// `containsAnalog`, and `hasUninferredWidth` bits, among others.
  RecursiveTypeProperties getRecursiveTypeProperties() const;

  //===--------------------------------------------------------------------===//
  // Convenience methods for accessing recursive type properties
  //===--------------------------------------------------------------------===//

  /// Returns true if this is or contains a 'const' type.
  bool containsConst() { return getRecursiveTypeProperties().containsConst; }

  /// Return true if this is or contains an Analog type.
  bool containsAnalog() { return getRecursiveTypeProperties().containsAnalog; }

  /// Return true if this is or contains a Reference type.
  bool containsReference() {
    return getRecursiveTypeProperties().containsReference;
  }

  /// Return true if this type contains an uninferred bit width.
  bool hasUninferredWidth() {
    return getRecursiveTypeProperties().hasUninferredWidth;
  }

  /// Return true if this type contains an uninferred bit reset.
  bool hasUninferredReset() {
    return getRecursiveTypeProperties().hasUninferredReset;
  }

  //===--------------------------------------------------------------------===//
  // Type classifications
  //===--------------------------------------------------------------------===//

  /// Return true if this is a 'ground' type, aka a non-aggregate type.
  bool isGround();

  /// Returns true if this is a 'const' type that can only hold compile-time
  /// constant values
  bool isConst();

protected:
  using Type::Type;
};

// Common base class for all base FIRRTL types.
class FIRRTLBaseType
    : public FIRRTLType::TypeBase<FIRRTLBaseType, FIRRTLType,
                                  detail::FIRRTLBaseTypeStorage> {
public:
  using Base::Base;

  /// Returns true if this is a 'const' type that can only hold compile-time
  /// constant values
  bool isConst();

  /// Return true if this is a "passive" type - one that contains no "flip"
  /// types recursively within itself.
  bool isPassive() const { return getRecursiveTypeProperties().isPassive; }

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLBaseType getPassiveType();

  /// Return a 'const' or non-'const' version of this type.
  FIRRTLBaseType getConstType(bool isConst);

  /// Return this type with all ground types replaced with UInt<1>.  This is
  /// used for `mem` operations.
  FIRRTLBaseType getMaskType();

  /// Return this type with widths of all ground types removed. This
  /// enables two types to be compared by structure and name ignoring
  /// widths.
  FIRRTLBaseType getWidthlessType();

  /// If this is an IntType, AnalogType, or sugar type for a single bit (Clock,
  /// Reset, etc) then return the bitwidth.  Return -1 if the is one of these
  /// types but without a specified bitwidth.  Return -2 if this isn't a simple
  /// type.
  int32_t getBitWidthOrSentinel();

  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return llvm::isa<FIRRTLDialect>(type.getDialect()) &&
           !type.isa<RefType, OpenBundleType, OpenVectorType, StringType>();
  }

  /// Returns true if this is a non-const "passive" that which is not analog.
  bool isRegisterType() {
    return isPassive() && !containsAnalog() && !containsConst();
  }

  /// Return true if this is a valid "reset" type.
  bool isResetType();

  //===--------------------------------------------------------------------===//
  // hw::FieldIDTypeInterface
  //===--------------------------------------------------------------------===//

  /// Get the maximum field ID of this type.  For integers and other ground
  /// types, there are no subfields and the maximum field ID is 0.  For bundle
  /// types and vector types, each field is assigned a field ID in a depth-first
  /// walk order. This function is used to calculate field IDs when this type is
  /// nested under another type.
  uint64_t getMaxFieldID();

  /// Get the sub-type of a type for a field ID, and the subfield's ID. Strip
  /// off a single layer of this type and return the sub-type and a field ID
  /// targeting the same field, but rebased on the sub-type.
  std::pair<circt::hw::FieldIDTypeInterface, uint64_t>
  getSubTypeByFieldID(uint64_t fieldID);

  /// Return the final type targeted by this field ID by recursively walking all
  /// nested aggregate types. This is the identity function for ground types.
  circt::hw::FieldIDTypeInterface getFinalTypeByFieldID(uint64_t fieldID);

  /// Returns the effective field id when treating the index field as the
  /// root of the type.  Essentially maps a fieldID to a fieldID after a
  /// subfield op. Returns the new id and whether the id is in the given
  /// child.
  std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID, uint64_t index);
};

/// Returns true if this is a 'const' type whose value is guaranteed to be
/// unchanging at circuit execution time
bool isConst(Type type);

/// Returns true if the type is or contains a 'const' type whose value is
/// guaranteed to be unchanging at circuit execution time
bool containsConst(Type type);

/// Returns whether the two types are equivalent.  This implements the exact
/// definition of type equivalence in the FIRRTL spec.  If the types being
/// compared have any outer flips that encode FIRRTL module directions (input or
/// output), these should be stripped before using this method.
bool areTypesEquivalent(FIRRTLType destType, FIRRTLType srcType,
                        bool destOuterTypeIsConst = false,
                        bool srcOuterTypeIsConst = false,
                        bool requireSameWidths = false);

/// Returns true if two types are weakly equivalent.  See the FIRRTL spec,
/// Section 4.6, for a full definition of this.  Roughly, the oriented types
/// (the types with any flips pushed to the leaves) must match.  This allows for
/// types with flips in different positions to be equivalent.
bool areTypesWeaklyEquivalent(FIRRTLType destType, FIRRTLType srcType,
                              bool destFlip = false, bool srcFlip = false,
                              bool destOuterTypeIsConst = false,
                              bool srcOuterTypeIsConst = false);

/// Returns whether the srcType can be const-casted to the destType.
bool areTypesConstCastable(FIRRTLType destType, FIRRTLType srcType,
                           bool srcOuterTypeIsConst = false);

/// Returns true if the destination is at least as wide as a source.  The source
/// and destination types must be equivalent non-analog types.  The types are
/// recursively connected to ensure that the destination is larger than the
/// source: ground types are compared on width, vector types are checked
/// recursively based on their elements and bundles are compared
/// field-by-field.  Types with unresolved widths are assumed to fit into or
/// hold their counterparts.
bool isTypeLarger(FIRRTLBaseType dstType, FIRRTLBaseType srcType);

mlir::Type getPassiveType(mlir::Type anyBaseFIRRTLType);

//===----------------------------------------------------------------------===//
// Width Qualified Ground Types
//===----------------------------------------------------------------------===//

/// Trait for types which have a width.
/// Users must implement:
/// ```c++
/// /// Return the width if known, or -1 if unknown.
/// int32_t getWidthOrSentinel();
/// ```
template <typename ConcreteType>
class WidthQualifiedTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, WidthQualifiedTypeTrait> {
public:
  /// Return an optional containing the width, if the width is known (or empty
  /// if width is unknown).
  std::optional<int32_t> getWidth() {
    auto width = static_cast<ConcreteType *>(this)->getWidthOrSentinel();
    if (width < 0)
      return std::nullopt;
    return width;
  }

  /// Return true if this integer type has a known width.
  bool hasWidth() {
    return 0 <= static_cast<ConcreteType *>(this)->getWidthOrSentinel();
  }
};

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

class SIntType;
class UIntType;

/// This is the common base class between SIntType and UIntType.
class IntType : public FIRRTLBaseType, public WidthQualifiedTypeTrait<IntType> {
public:
  using FIRRTLBaseType::FIRRTLBaseType;

  /// Return an SIntType or UIntType with the specified signedness, width, and
  /// constness.
  static IntType get(MLIRContext *context, bool isSigned,
                     int32_t widthOrSentinel = -1, bool isConst = false);

  bool isSigned() { return isa<SIntType>(); }
  bool isUnsigned() { return isa<UIntType>(); }

  /// Return the width of this type, or -1 if it has none specified.
  int32_t getWidthOrSentinel();

  /// Return a 'const' or non-'const' version of this type.
  IntType getConstType(bool isConst);

  static bool classof(Type type) {
    return type.isa<SIntType>() || type.isa<UIntType>();
  }
};

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

// Get the bit width for this type, return None  if unknown. Unlike
// getBitWidthOrSentinel(), this can recursively compute the bitwidth of
// aggregate types. For bundle and vectors, recursively get the width of each
// field element and return the total bit width of the aggregate type. This
// returns None, if any of the bundle fields is a flip type, or ground type with
// unknown bit width.
std::optional<int64_t> getBitWidth(FIRRTLBaseType type,
                                   bool ignoreFlip = false);

// Parse a FIRRTL type without a leading `!firrtl.` dialect tag.
ParseResult parseNestedType(FIRRTLType &result, AsmParser &parser);
ParseResult parseNestedBaseType(FIRRTLBaseType &result, AsmParser &parser);

// Print a FIRRTL type without a leading `!firrtl.` dialect tag.
void printNestedType(Type type, AsmPrinter &os);

using FIRRTLValue = mlir::TypedValue<FIRRTLType>;
using FIRRTLBaseValue = mlir::TypedValue<FIRRTLBaseType>;

} // namespace firrtl
} // namespace circt

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h.inc"

namespace llvm {

// Type hash just like pointers.
template <>
struct DenseMapInfo<circt::firrtl::FIRRTLType> {
  using FIRRTLType = circt::firrtl::FIRRTLType;
  static FIRRTLType getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return FIRRTLType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static FIRRTLType getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return FIRRTLType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(FIRRTLType val) { return mlir::hash_value(val); }
  static bool isEqual(FIRRTLType LHS, FIRRTLType RHS) { return LHS == RHS; }
};

} // namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_TYPES_H
