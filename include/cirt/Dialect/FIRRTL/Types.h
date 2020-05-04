//===- FIRRTL/IR/Ops.h - FIRRTL dialect -------------------------*- C++ -*-===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRT_DIALECT_FIRRTL_IR_TYPES_H
#define CIRT_DIALECT_FIRRTL_IR_TYPES_H

#include "mlir/IR/Types.h"

namespace cirt {
namespace firrtl {
namespace detail {
struct WidthTypeStorage;
struct FlipTypeStorage;
struct BundleTypeStorage;
struct VectorTypeStorage;
} // namespace detail.

using namespace mlir;

// This is a common base class for all FIRRTL types.
class FIRRTLType : public Type {
public:
  enum Kind {
    FIRST_KIND = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,

    // Ground Types Without Parameters.
    Clock = FIRST_KIND,
    Reset,
    AsyncReset,

    // Width Qualified Ground Types.
    SInt,
    UInt,
    Analog,

    // Derived Types
    Flip,
    Bundle,
    Vector,
    LAST_KIND = Vector
  };

  void print(raw_ostream &os) const;

  /// Return true if this is a "passive" type - one that contains no "flip"
  /// types recursively within itself.
  bool isPassiveType();

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLType getPassiveType();

  /// Return this type with all ground types replaced with UInt<1>.  This is
  /// used for `mem` operations.
  FIRRTLType getMaskType();

  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return type.getKind() >= Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE &&
           type.getKind() <= FIRRTLType::LAST_KIND;
  }

  static bool kindof(unsigned kind) {
    return kind >= FIRST_KIND && kind <= LAST_KIND;
  }

protected:
  using Type::Type;
};

//===----------------------------------------------------------------------===//
// Ground Types Without Parameters
//===----------------------------------------------------------------------===//

/// `firrtl.Clock` describe wires and ports meant for carrying clock signals.
class ClockType : public FIRRTLType::TypeBase<ClockType, FIRRTLType> {
public:
  using Base::Base;
  static ClockType get(MLIRContext *context) {
    return Base::get(context, FIRRTLType::Clock);
  }
  static bool kindof(unsigned kind) { return kind == Clock; }
};

/// `firrtl.Reset`.
/// TODO(firrtl spec): This is not described in the FIRRTL spec.
class ResetType : public FIRRTLType::TypeBase<ResetType, FIRRTLType> {
public:
  using Base::Base;
  static ResetType get(MLIRContext *context) {
    return Base::get(context, FIRRTLType::Reset);
  }
  static bool kindof(unsigned kind) { return kind == Reset; }
};

/// `firrtl.AsyncReset`.
/// TODO(firrtl spec): This is not described in the FIRRTL spec.
class AsyncResetType : public FIRRTLType::TypeBase<AsyncResetType, FIRRTLType> {
public:
  using Base::Base;
  static AsyncResetType get(MLIRContext *context) {
    return Base::get(context, FIRRTLType::AsyncReset);
  }
  static bool kindof(unsigned kind) { return kind == AsyncReset; }
};

//===----------------------------------------------------------------------===//
// Width Qualified Ground Types
//===----------------------------------------------------------------------===//

template <typename ConcreteType, FIRRTLType::Kind typeKind, typename ParentType>
class WidthQualifiedType
    : public FIRRTLType::TypeBase<ConcreteType, ParentType,
                                  detail::WidthTypeStorage> {
public:
  using FIRRTLType::TypeBase<ConcreteType, ParentType,
                             detail::WidthTypeStorage>::Base::Base;

  /// Return the width of this type, or -1 if it has none specified.
  int32_t getWidthOrSentinel() {
    auto width = static_cast<ConcreteType *>(this)->getWidth();
    return width.hasValue() ? width.getValue() : -1;
  }

  static bool kindof(unsigned kind) { return kind == typeKind; }
};

/// This is the common base class between SIntType and UIntType.
class IntType : public FIRRTLType {
public:
  using FIRRTLType::FIRRTLType;

  /// Return a SIntType or UInt type with the specified signedness and width.
  static IntType get(MLIRContext *context, bool isSigned, int32_t width = -1);

  bool isSigned() { return getKind() == SInt; }
  bool isUnsigned() { return getKind() == UInt; }

  /// Return true if this integer type has a known width.
  bool hasWidth() { return getWidth().hasValue(); }

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth();

  /// Return the width of this type, or -1 if it has none specified.
  int32_t getWidthOrSentinel() {
    auto width = getWidth();
    return width.hasValue() ? width.getValue() : -1;
  }

  static bool kindof(unsigned kind) { return kind == SInt || kind == UInt; }
  static bool classof(Type type) { return kindof(type.getKind()); }
};

/// A signed integer type, whose width may not be known.
class SIntType
    : public WidthQualifiedType<SIntType, FIRRTLType::SInt, IntType> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static SIntType get(MLIRContext *context, int32_t width = -1);

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth();
};

/// An unsigned integer type, whose width may not be known.
class UIntType
    : public WidthQualifiedType<UIntType, FIRRTLType::UInt, IntType> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static UIntType get(MLIRContext *context, int32_t width = -1);

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth();
};

// `firrtl.Analog` can be attached to multiple drivers.
class AnalogType
    : public WidthQualifiedType<AnalogType, FIRRTLType::Analog, FIRRTLType> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static AnalogType get(MLIRContext *context, int32_t width = -1);

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth();
};

//===----------------------------------------------------------------------===//
// Flip Type
//===----------------------------------------------------------------------===//

class FlipType : public FIRRTLType::TypeBase<FlipType, FIRRTLType,
                                             detail::FlipTypeStorage> {
public:
  using Base::Base;

  FIRRTLType getElementType();

  static FIRRTLType get(FIRRTLType element);

  static bool kindof(unsigned kind) { return kind == Flip; }
};

//===----------------------------------------------------------------------===//
// Bundle Type
//===----------------------------------------------------------------------===//

/// BundleType is an aggregate of named elements.  This is effectively a struct
/// for FIRRTL.
class BundleType : public FIRRTLType::TypeBase<BundleType, FIRRTLType,
                                               detail::BundleTypeStorage> {
public:
  using Base::Base;

  // Each element of a bundle, which is a name and type.
  using BundleElement = std::pair<Identifier, FIRRTLType>;

  static FIRRTLType get(ArrayRef<BundleElement> elements, MLIRContext *context);

  ArrayRef<BundleElement> getElements();

  size_t getNumElements() { return getElements().size(); }

  /// Look up an element by name.  This returns None on failure.
  llvm::Optional<BundleElement> getElement(StringRef name);
  FIRRTLType getElementType(StringRef name);

  /// Return true if this is a "passive" type - one that contains no "flip"
  /// types recursively within itself.
  bool isPassiveType();

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLType getPassiveType();

  static bool kindof(unsigned kind) { return kind == Bundle; }
};

//===----------------------------------------------------------------------===//
// FVector Type
//===----------------------------------------------------------------------===//

/// VectorType is a fixed size collection of elements, like an array.
class FVectorType : public FIRRTLType::TypeBase<FVectorType, FIRRTLType,
                                                detail::VectorTypeStorage> {
public:
  using Base::Base;

  static FIRRTLType get(FIRRTLType elementType, unsigned numElements);

  FIRRTLType getElementType();
  unsigned getNumElements();

  /// Return true if this is a "passive" type - one that contains no "flip"
  /// types recursively within itself.
  bool isPassiveType();

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLType getPassiveType();

  static bool kindof(unsigned kind) { return kind == Vector; }
};

} // namespace firrtl
} // namespace cirt

namespace llvm {

// Type hash just like pointers.
template <>
struct DenseMapInfo<cirt::firrtl::FIRRTLType> {
  using FIRRTLType = cirt::firrtl::FIRRTLType;
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

#endif // CIRT_DIALECT_FIRRTL_IR_TYPES_H