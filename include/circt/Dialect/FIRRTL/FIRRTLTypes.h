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
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace firrtl {
namespace detail {
struct WidthTypeStorage;
struct BundleTypeStorage;
struct VectorTypeStorage;
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
class FVectorType;
class RefType;

/// A collection of bits indicating the recursive properties of a type.
struct RecursiveTypeProperties {
  /// Whether the type only contains passive elements.
  bool isPassive : 1;
  /// Whether the type contains an analog type.
  bool containsAnalog : 1;
  /// Whether the type has any uninferred bit widths.
  bool hasUninferredWidth : 1;

  /// The number of bits required to represent a type's recursive properties.
  static constexpr unsigned numBits = 3;
  /// Unpack `RecursiveTypeProperties` from a bunch of bits.
  static RecursiveTypeProperties fromFlags(unsigned bits);
  /// Pack `RecursiveTypeProperties` as a bunch of bits.
  unsigned toFlags() const;
};

// This is a common base class for all FIRRTL types.
class FIRRTLType : public Type {
public:
  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return llvm::isa<FIRRTLDialect>(type.getDialect());
  }

protected:
  using Type::Type;
};

// Common base class for all base FIRRTL types.
class FIRRTLBaseType : public FIRRTLType {
public:
  /// Return true if this is a "passive" type - one that contains no "flip"
  /// types recursively within itself.
  bool isPassive() { return getRecursiveTypeProperties().isPassive; }

  /// Returns true if this is a "passive" that which is not analog.
  bool isRegisterType() { return isPassive() && !containsAnalog(); }

  /// Return true if this is a 'ground' type, aka a non-aggregate type.
  bool isGround();

  /// Return true if this is or contains an Analog type.
  bool containsAnalog() { return getRecursiveTypeProperties().containsAnalog; }

  /// Return true if this type contains an uninferred bit width.
  bool hasUninferredWidth() {
    return getRecursiveTypeProperties().hasUninferredWidth;
  }

  /// Return the recursive properties of the type, containing the `isPassive`,
  /// `containsAnalog`, and `hasUninferredWidth` bits.
  RecursiveTypeProperties getRecursiveTypeProperties();

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLBaseType getPassiveType();

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
    return llvm::isa<FIRRTLDialect>(type.getDialect()) && !type.isa<RefType>();
  }

  /// Return true if this is a valid "reset" type.
  bool isResetType();

  /// Get the maximum field ID of this type.  For integers and other ground
  /// types, there are no subfields and the maximum field ID is 0.  For bundle
  /// types and vector types, each field is assigned a field ID in a depth-first
  /// walk order. This function is used to calculate field IDs when this type is
  /// nested under another type.
  uint64_t getMaxFieldID();

  /// Get the sub-type of a type for a field ID, and the subfield's ID. Strip
  /// off a single layer of this type and return the sub-type and a field ID
  /// targeting the same field, but rebased on the sub-type.
  std::pair<FIRRTLBaseType, uint64_t> getSubTypeByFieldID(uint64_t fieldID);

  /// Return the final type targeted by this field ID by recursively walking all
  /// nested aggregate types. This is the identity function for ground types.
  FIRRTLBaseType getFinalTypeByFieldID(uint64_t fieldID);

  /// Returns the effective field id when treating the index field as the
  /// root of the type.  Essentially maps a fieldID to a fieldID after a
  /// subfield op. Returns the new id and whether the id is in the given
  /// child.
  std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID, uint64_t index);

  /// Get the number of ground (non-aggregate) fields in the type.  A field
  /// which is a bundle or vector is not counted, but the recursive ground
  /// fields of are.
  uint64_t getGroundFields() const;

protected:
  using FIRRTLType::FIRRTLType;
};

/// Returns whether the two types are equivalent.  This implements the exact
/// definition of type equivalence in the FIRRTL spec.  If the types being
/// compared have any outer flips that encode FIRRTL module directions (input or
/// output), these should be stripped before using this method.
bool areTypesEquivalent(FIRRTLType destType, FIRRTLType srcType);

/// Returns true if two types are weakly equivalent.  See the FIRRTL spec,
/// Section 4.6, for a full definition of this.  Roughly, the oriented types
/// (the types with any flips pushed to the leaves) must match.  This allows for
/// types with flips in different positions to be equivalent.
bool areTypesWeaklyEquivalent(FIRRTLType destType, FIRRTLType srcType,
                              bool destFlip = false, bool srcFlip = false);

/// Returns true if the destination is at least as wide as a source.  The source
/// and destination types must be equivalent non-analog types.  The types are
/// recursively connected to ensure that the destination is larger than the
/// source: ground types are compared on width, vector types are checked
/// recursively based on their elements and bundles are compared
/// field-by-field.  Types with unresolved widths are assumed to fit into or
/// hold their counterparts.
bool isTypeLarger(FIRRTLBaseType dstType, FIRRTLBaseType srcType);

mlir::Type getVectorElementType(mlir::Type array);
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

  /// Return an SIntType or UIntType with the specified signedness and width.
  static IntType get(MLIRContext *context, bool isSigned,
                     int32_t widthOrSentinel = -1);

  bool isSigned() { return isa<SIntType>(); }
  bool isUnsigned() { return isa<UIntType>(); }

  /// Return the width of this type, or -1 if it has none specified.
  int32_t getWidthOrSentinel();

  static bool classof(Type type) {
    return type.isa<SIntType>() || type.isa<UIntType>();
  }
};

//===----------------------------------------------------------------------===//
// Reference Type
//===----------------------------------------------------------------------===//

class RefType
    : public FIRRTLType::TypeBase<RefType, FIRRTLType, detail::RefTypeStorage> {
public:
  using Base::Base;
  static RefType get(FIRRTLBaseType type);

  /// Return the underlying type.
  FIRRTLBaseType getType();

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitErrorFn,
                              FIRRTLBaseType base);
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

template <>
struct DenseMapInfo<circt::firrtl::FIRRTLBaseType> {
  using FIRRTLBaseType = circt::firrtl::FIRRTLBaseType;
  static FIRRTLBaseType getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return FIRRTLBaseType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static FIRRTLBaseType getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return FIRRTLBaseType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(FIRRTLBaseType val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(FIRRTLBaseType LHS, FIRRTLBaseType RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_TYPES_H
