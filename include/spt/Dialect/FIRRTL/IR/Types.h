//===- FIRRTL/IR/Ops.h - FIRRTL dialect -------------------------*- C++ -*-===//
//
// This file defines an MLIR dialect for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_IR_TYPES_H
#define SPT_DIALECT_FIRRTL_IR_TYPES_H

#include "mlir/IR/Types.h"

namespace spt {
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

    // Width Qualified Ground Types.
    SInt,
    UInt,
    Analog,

    // Derived Types
    Flip,
    Bundle,
    Vector,
    LAST_KIND
  };

  void print(raw_ostream &os) const;

  static bool kindof(unsigned kind) {
    return kind >= FIRST_KIND && kind < LAST_KIND;
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

/// `firrtl.Reset`.  FIXME: This is not described in the FIRRTL spec, nor is
/// AsyncReset.
class ResetType : public FIRRTLType::TypeBase<ResetType, FIRRTLType> {
public:
  using Base::Base;
  static ResetType get(MLIRContext *context) {
    return Base::get(context, FIRRTLType::Reset);
  }
  static bool kindof(unsigned kind) { return kind == Reset; }
};

//===----------------------------------------------------------------------===//
// Width Qualified Ground Types
//===----------------------------------------------------------------------===//

namespace detail {
Optional<int32_t> getWidthQualifiedTypeWidth(WidthTypeStorage *impl);
} // namespace detail.

template <typename ConcreteType, FIRRTLType::Kind typeKind>
class WidthQualifiedType
    : public FIRRTLType::TypeBase<ConcreteType, FIRRTLType,
                                  detail::WidthTypeStorage> {
public:
  using FIRRTLType::TypeBase<ConcreteType, FIRRTLType,
                             detail::WidthTypeStorage>::Base::Base;

  static bool kindof(unsigned kind) { return kind == typeKind; }

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth() const {
    return getWidthQualifiedTypeWidth(this->getImpl());
  }
};

/// A signed integer type, whose width may not be known.
class SIntType : public WidthQualifiedType<SIntType, FIRRTLType::SInt> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static SIntType get(MLIRContext *context, int32_t width = -1);
};

/// An unsigned integer type, whose width may not be known.
class UIntType : public WidthQualifiedType<UIntType, FIRRTLType::UInt> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static UIntType get(MLIRContext *context, int32_t width = -1);
};

// `firrtl.Analog` can be attached to multiple drivers.
class AnalogType : public WidthQualifiedType<AnalogType, FIRRTLType::Analog> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static AnalogType get(MLIRContext *context, int32_t width = -1);
};

//===----------------------------------------------------------------------===//
// Flip Type
//===----------------------------------------------------------------------===//

class FlipType : public FIRRTLType::TypeBase<FlipType, FIRRTLType,
                                             detail::FlipTypeStorage> {
public:
  using Base::Base;

  FIRRTLType getElementType() const;

  static FlipType get(FIRRTLType element);

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

  static BundleType get(ArrayRef<BundleElement> elements, MLIRContext *context);

  ArrayRef<BundleElement> getElements() const;

  static bool kindof(unsigned kind) { return kind == Bundle; }
};

//===----------------------------------------------------------------------===//
// Vector Type
//===----------------------------------------------------------------------===//

/// VectorType is a fixed size collection of elements, like an array.
class FVectorType : public FIRRTLType::TypeBase<FVectorType, FIRRTLType,
                                               detail::VectorTypeStorage> {
public:
  using Base::Base;

  static FVectorType get(FIRRTLType elementType, unsigned numElements);

  FIRRTLType getElementType() const;
  unsigned getNumElements() const;

  static bool kindof(unsigned kind) { return kind == Vector; }
};

} // namespace firrtl
} // namespace spt

namespace llvm {

// Type hash just like pointers.
template <>
struct DenseMapInfo<spt::firrtl::FIRRTLType> {
  using FIRRTLType = spt::firrtl::FIRRTLType;
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

#endif // SPT_DIALECT_FIRRTL_IR_TYPES_H