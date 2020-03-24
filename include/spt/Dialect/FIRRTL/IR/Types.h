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
// Ground Types
//===----------------------------------------------------------------------===//

/// A signed integer type, whose width may not be known.
class SIntType : public Type::TypeBase<SIntType, Type> {
public:
  using Base::Base;

  static SIntType get(MLIRContext *context) {
    return Base::get(context, FIRRTLTypes::SInt);
  }
  static bool kindof(unsigned kind) { return kind == FIRRTLTypes::SInt; }
};

/// A unsigned integer type with unknown width.
class UIntType : public Type::TypeBase<UIntType, Type> {
public:
  using Base::Base;
  static UIntType get(MLIRContext *context) {
    return Base::get(context, FIRRTLTypes::UInt);
  }
  static bool kindof(unsigned kind) { return kind == FIRRTLTypes::UInt; }
};

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
// Derived Types
//===----------------------------------------------------------------------===//

namespace detail {
struct AnalogTypeStorage;
} // namespace detail.

// `firrtl.Analog` can be attached to multiple drivers.
class AnalogType
    : public Type::TypeBase<AnalogType, Type, detail::AnalogTypeStorage> {
public:
  using Base::Base;

  /// Get an AnalogType with a known width, or -1 for unknown.
  static AnalogType get(int32_t width, MLIRContext *context);

  /// Get an AnalogType with unknown width.
  static AnalogType get(MLIRContext *context) { return get(-1, context); }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == FIRRTLTypes::Analog; }

  /// Return the bitwidth of this Analog type or None if unknown.
  Optional<int32_t> getWidth() const;
};

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_IR_TYPES_H