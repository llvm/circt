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
using namespace mlir;

namespace FIRRTLTypes {
enum Kind {
  // Ground types.
  SInt = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  UInt,
  Clock,
  Reset,
  // Derived types.
  Analog,
};
} // namespace FIRRTLTypes

//===----------------------------------------------------------------------===//
// Ground Types Without Parameters
//===----------------------------------------------------------------------===//

/// `firrtl.Clock` describe wires and ports meant for carrying clock signals.
class ClockType : public Type::TypeBase<ClockType, Type> {
public:
  using Base::Base;
  static ClockType get(MLIRContext *context) {
    return Base::get(context, FIRRTLTypes::Clock);
  }
  static bool kindof(unsigned kind) { return kind == FIRRTLTypes::Clock; }
};

/// `firrtl.Reset`.  FIXME: This is not described in the FIRRTL spec, nor is
/// AsyncReset.
class ResetType : public Type::TypeBase<ResetType, Type> {
public:
  using Base::Base;
  static ResetType get(MLIRContext *context) {
    return Base::get(context, FIRRTLTypes::Reset);
  }
  static bool kindof(unsigned kind) { return kind == FIRRTLTypes::Reset; }
};

//===----------------------------------------------------------------------===//
// Width Qualified Types
//===----------------------------------------------------------------------===//

namespace detail {
struct WidthTypeStorage;
Optional<int32_t> getWidthQualifiedTypeWidth(WidthTypeStorage *impl);
} // namespace detail.

template <typename ConcreteType, FIRRTLTypes::Kind typeKind>
class WidthQualifiedType
    : public Type::TypeBase<ConcreteType, Type, detail::WidthTypeStorage> {
public:
  using Type::TypeBase<ConcreteType, Type,
                       detail::WidthTypeStorage>::Base::Base;

  static bool kindof(unsigned kind) { return kind == typeKind; }

  /// Return the bitwidth of this type or None if unknown.
  Optional<int32_t> getWidth() const {
    return getWidthQualifiedTypeWidth(this->getImpl());
  }
};

/// A signed integer type, whose width may not be known.
class SIntType : public WidthQualifiedType<SIntType, FIRRTLTypes::SInt> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static SIntType get(MLIRContext *context, int32_t width = -1);
};

/// An unsigned integer type, whose width may not be known.
class UIntType : public WidthQualifiedType<UIntType, FIRRTLTypes::UInt> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static UIntType get(MLIRContext *context, int32_t width = -1);
};

// `firrtl.Analog` can be attached to multiple drivers.
class AnalogType : public WidthQualifiedType<AnalogType, FIRRTLTypes::Analog> {
public:
  using WidthQualifiedType::WidthQualifiedType;

  /// Get an with a known width, or -1 for unknown.
  static AnalogType get(MLIRContext *context, int32_t width = -1);
};

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_IR_TYPES_H