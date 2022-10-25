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
  unsigned getMaxFieldID();

  /// Get the sub-type of a type for a field ID, and the subfield's ID. Strip
  /// off a single layer of this type and return the sub-type and a field ID
  /// targeting the same field, but rebased on the sub-type.
  std::pair<FIRRTLBaseType, unsigned> getSubTypeByFieldID(unsigned fieldID);

  /// Return the final type targeted by this field ID by recursively walking all
  /// nested aggregate types. This is the identity function for ground types.
  FIRRTLBaseType getFinalTypeByFieldID(unsigned fieldID);

  /// Returns the effective field id when treating the index field as the
  /// root of the type.  Essentially maps a fieldID to a fieldID after a
  /// subfield op. Returns the new id and whether the id is in the given
  /// child.
  std::pair<unsigned, bool> rootChildFieldID(unsigned fieldID, unsigned index);

  /// Get the number of ground (non-aggregate) fields in the type.  A field
  /// which is a bundle or vector is not counted, but the recursive ground
  /// fields of are.
  unsigned getGroundFields() const;

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

template <typename ConcreteType>
class WidthQualifiedTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, WidthQualifiedTrait> {
public:
  Optional<int32_t> getWidth() {
    auto v = static_cast<ConcreteType *>(this)->getBaseWidth();
    if (v >= 0)
      return v;
    return {};
  }
  int32_t getWidthOrSentinel() {
    return static_cast<ConcreteType *>(this)->getBaseWidth();
  }
  bool hasWidth() {
    return static_cast<ConcreteType *>(this)->getBaseWidth() >= 0;
  }
  ConcreteType changeWidth(int32_t width) {
    return ConcreteType::get(static_cast<ConcreteType *>(this)->getContext(),
                             width);
  }
};

template <typename ConcreteType, typename ParentType>
class WidthQualifiedType
    : public FIRRTLType::TypeBase<ConcreteType, ParentType,
                                  detail::WidthTypeStorage> {
public:
  using FIRRTLType::TypeBase<ConcreteType, ParentType,
                             detail::WidthTypeStorage>::Base::Base;

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth() {
    return static_cast<ConcreteType *>(this)->getWidth();
  }

  /// Return the width of this type, or -1 if it has none specified.
  int32_t getWidthOrSentinel() { return getWidth().value_or(-1); }

  /// Return true if this type has a known width.
  bool hasWidth() { return getWidth().has_value(); }

  /// Return a new type with the width changed to a different value.
  ConcreteType changeWidth(int32_t width) {
    return ConcreteType::get(static_cast<ConcreteType *>(this)->getContext(),
                             width);
  }
};

class SIntType;
class UIntType;

/// This is the common base class between SIntType and UIntType.
class IntType : public FIRRTLBaseType {
public:
  using FIRRTLBaseType::FIRRTLBaseType;

  /// Return a SIntType or UInt type with the specified signedness and width.
  static IntType get(MLIRContext *context, bool isSigned, int32_t width = -1);

  bool isSigned() { return isa<SIntType>(); }
  bool isUnsigned() { return isa<UIntType>(); }

  /// Return true if this integer type has a known width.
  bool hasWidth() { return getWidth().has_value(); }

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth();

  /// Return the width of this type, or -1 if it has none specified.
  int32_t getWidthOrSentinel() { return getWidth().value_or(-1); }

  static bool classof(Type type) {
    return type.isa<SIntType>() || type.isa<UIntType>();
  }
};

/// A signed integer type, whose width may not be known.
class SIntType : public WidthQualifiedType<SIntType, IntType> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static SIntType get(MLIRContext *context, int32_t width = -1);

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth();
};

/// An unsigned integer type, whose width may not be known.
class UIntType : public WidthQualifiedType<UIntType, IntType> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static UIntType get(MLIRContext *context, int32_t width = -1);

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth();
};

//===----------------------------------------------------------------------===//
// Bundle Type
//===----------------------------------------------------------------------===//

/// BundleType is an aggregate of named elements.  This is effectively a struct
/// for FIRRTL.
class BundleType : public FIRRTLType::TypeBase<BundleType, FIRRTLBaseType,
                                               detail::BundleTypeStorage> {
public:
  using Base::Base;

  // Each element of a bundle, which is a name and type.
  struct BundleElement {
    StringAttr name;
    bool isFlip;
    FIRRTLBaseType type;

    BundleElement(StringAttr name, bool isFlip, FIRRTLBaseType type)
        : name(name), isFlip(isFlip), type(type) {}

    bool operator==(const BundleElement &rhs) const {
      return name == rhs.name && isFlip == rhs.isFlip && type == rhs.type;
    }
    bool operator!=(const BundleElement &rhs) const { return !operator==(rhs); }
  };

  static BundleType get(ArrayRef<BundleElement> elements, MLIRContext *context);

  ArrayRef<BundleElement> getElements() const;

  size_t getNumElements() { return getElements().size(); }

  /// Look up an element's index by name.  This returns None on failure.
  llvm::Optional<unsigned> getElementIndex(StringAttr name);
  llvm::Optional<unsigned> getElementIndex(StringRef name);

  /// Look up an element's name by index. This asserts if index is invalid.
  StringRef getElementName(size_t index);

  /// Look up an element by name.  This returns None on failure.
  llvm::Optional<BundleElement> getElement(StringAttr name);
  llvm::Optional<BundleElement> getElement(StringRef name);

  /// Look up an element by index.  This asserts if index is invalid.
  BundleElement getElement(size_t index);

  /// Look up an element type by name.
  FIRRTLBaseType getElementType(StringAttr name);
  FIRRTLBaseType getElementType(StringRef name);

  /// Look up an element type by index.
  FIRRTLBaseType getElementType(size_t index);

  /// Return the recursive properties of the type.
  RecursiveTypeProperties getRecursiveTypeProperties();

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLBaseType getPassiveType();

  /// Get an integer ID for the field. Field IDs start at 1, and are assigned
  /// to each field in a bundle in a recursive pre-order walk of all fields,
  /// visiting all nested bundle fields.  A field ID of 0 is used to reference
  /// the bundle itself. The ID can be used to uniquely identify any specific
  /// field in this bundle.
  unsigned getFieldID(unsigned index);

  /// Find the element index corresponding to the desired fieldID.  If the
  /// fieldID corresponds to a field in a nested bundle, it will return the
  /// index of the parent field.
  unsigned getIndexForFieldID(unsigned fieldID);

  /// Strip off a single layer of this type and return the sub-type and a field
  /// ID targeting the same field, but rebased on the sub-type.
  std::pair<FIRRTLBaseType, unsigned> getSubTypeByFieldID(unsigned fieldID);

  /// Get the maximum field ID in this bundle.  This is helpful for constructing
  /// field IDs when this BundleType is nested in another aggregate type.
  unsigned getMaxFieldID();

  /// Returns the effective field id when treating the index field as the root
  /// of the type.  Essentially maps a fieldID to a fieldID after a subfield op.
  /// Returns the new id and whether the id is in the given child.
  std::pair<unsigned, bool> rootChildFieldID(unsigned fieldID, unsigned index);

  using iterator = ArrayRef<BundleElement>::iterator;
  iterator begin() const { return getElements().begin(); }
  iterator end() const { return getElements().end(); }
};

//===----------------------------------------------------------------------===//
// FVector Type
//===----------------------------------------------------------------------===//

/// VectorType is a fixed size collection of elements, like an array.
class FVectorType : public FIRRTLType::TypeBase<FVectorType, FIRRTLBaseType,
                                                detail::VectorTypeStorage> {
public:
  using Base::Base;

  static FVectorType get(FIRRTLBaseType elementType, size_t numElements);

  FIRRTLBaseType getElementType();
  size_t getNumElements();

  /// Return the recursive properties of the type.
  RecursiveTypeProperties getRecursiveTypeProperties();

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLBaseType getPassiveType();

  /// Get an integer ID for the field. Field IDs start at 1, and are assigned
  /// to each field in a vector in a recursive depth-first walk of all elements.
  /// A field ID of 0 is used to reference the vector itself.
  size_t getFieldID(size_t index);

  /// Find the element index corresponding to the desired fieldID.  If the
  /// fieldID corresponds to a field in nested under an element, it will return
  /// the index of the parent element.
  size_t getIndexForFieldID(size_t fieldID);

  /// Strip off a single layer of this type and return the sub-type and a field
  /// ID targeting the same field, but rebased on the sub-type.
  std::pair<FIRRTLBaseType, size_t> getSubTypeByFieldID(size_t fieldID);

  /// Get the maximum field ID in this vector.  This is helpful for constructing
  /// field IDs when this VectorType is nested in another aggregate type.
  size_t getMaxFieldID();

  /// Returns the effective field id when treating the index field as the root
  /// of the type.  Essentially maps a fieldID to a fieldID after a subfield op.
  /// Returns the new id and whether the id is in the given child.
  std::pair<size_t, bool> rootChildFieldID(size_t fieldID, size_t index);
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
llvm::Optional<int64_t> getBitWidth(FIRRTLBaseType type,
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
