//===- MooreTypes.h - Declare Moore dialect types ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORETYPES_H
#define CIRCT_DIALECT_MOORE_MOORETYPES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include <variant>

namespace circt {
namespace moore {

class ArrayType;
class AssocArrayType;
class ChandleType;
class EventType;
class IntType;
class OpenArrayType;
class OpenUnpackedArrayType;
class PackedType;
class QueueType;
class RealType;
class StringType;
class StructType;
class TimeType;
class UnionType;
class UnpackedType;
class UnpackedArrayType;
class UnpackedStructType;
class UnpackedUnionType;
class VoidType;

/// The number of values each bit of a type can assume.
enum class Domain {
  /// Two-valued types such as `bit` or `int`.
  TwoValued,
  /// Four-valued types such as `logic` or `integer`.
  FourValued,
};

/// The type of floating point / real number behind a RealType
enum class RealWidth {
  /// A standard 32-Bit floating point number ("float")
  f32 = 32,
  /// A 64-bit double-precision floation point number ("double")
  f64 = 64
};


/// Check if a type is an `IntType` type of the given width.
bool isIntType(Type type, unsigned width);
/// Check if a type is an `IntType` type of the given domain.
bool isIntType(Type type, Domain domain);
/// Check if a type is an `IntType` type of the given width and domain.
bool isIntType(Type type, unsigned width, Domain domain);
/// Check if a type is a `RealType` type of the given width.
bool isRealType(Type type, unsigned width);

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

/// An unpacked SystemVerilog type.
///
/// Unpacked types are a second level of types in SystemVerilog. They extend a
/// core unpacked type with a variety of unpacked dimensions, depending on which
/// syntactic construct generated the type (variable or otherwise). The core
/// unpacked types are:
///
/// - Packed types
/// - Non-integer types: `shortreal`, `real`, `realtime`
/// - Unpacked structs and unions
/// - `string`, `chandle`, `event`
/// - Virtual interfaces
/// - Class types
/// - Covergroups
/// - Unpacked named types
/// - Unpacked type references
///
/// The unpacked dimensions are:
///
/// - Unsized (`[]`)
/// - Arrays (`[x]`)
/// - Ranges (`[x:y]`)
/// - Associative (`[T]` or `[*]`)
/// - Queues (`[$]` or `[$:x]`)
class UnpackedType : public Type {
public:
  static bool classof(Type type) {
    return llvm::isa<PackedType, StringType, ChandleType, EventType, RealType,
                     UnpackedArrayType, OpenUnpackedArrayType, AssocArrayType,
                     QueueType, UnpackedStructType, UnpackedUnionType>(type);
  }

  /// Get the value domain of this type.
  Domain getDomain() const;

  /// Get the size of this type in bits.
  ///
  /// Returns `None` if any of the type's dimensions is unsized, associative, or
  /// a queue, or the core type itself has no known size.
  std::optional<unsigned> getBitSize() const;

  // Support parsing and printing of unpacked types in their prefix-stripped
  // form.
  static Type parse(mlir::AsmParser &odsParser);
  void print(mlir::AsmPrinter &odsPrinter) const;

protected:
  using Type::Type;
};

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

/// A packed SystemVerilog type.
///
/// Packed types are the core types of SystemVerilog. They combine a core packed
/// type with an optional sign and zero or more packed dimensions. The core
/// packed types are:
///
/// - Integer vector types: `bit`, `logic`, `reg`
/// - Integer atom types: `byte`, `shortint`, `int`, `longint`, `integer`,
///   `time`
/// - Packed structs and unions
/// - Enums
/// - Packed named types
/// - Packed type references
///
/// The packed dimensions can be:
///
/// - Unsized (`[]`)
/// - Ranges (`[x:y]`)
///
/// Note that every packed type is also a valid unpacked type. But unpacked
/// types are *not* valid packed types.
class PackedType : public UnpackedType {
public:
  static bool classof(Type type) {
    return llvm::isa<VoidType, IntType, ArrayType, OpenArrayType, StructType,
                     UnionType, TimeType>(type);
  }

  /// Get the value domain of this type.
  Domain getDomain() const;

  /// Get the size of this type in bits.
  ///
  /// Returns `None` if any of the type's dimensions is unsized.
  std::optional<unsigned> getBitSize() const;

  /// Get the simple bit vector type equivalent to this packed type. Returns
  /// null if the type does not have a known bit size.
  IntType getSimpleBitVector() const;

  /// Check if this is a `TimeType`, or an aggregate that contains a nested
  /// `TimeType`.
  bool containsTimeType() const;

protected:
  using UnpackedType::UnpackedType;
};

//===----------------------------------------------------------------------===//
// Struct Members
//===----------------------------------------------------------------------===//

/// A member of a struct.
struct StructLikeMember {
  /// The name of this member.
  StringAttr name;
  /// The type of this member.
  UnpackedType type;

  bool operator==(const StructLikeMember &other) const {
    return name == other.name && type == other.type;
  }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const StructLikeMember &x) {
  return llvm::hash_combine(x.name, x.type);
}

} // namespace moore
} // namespace circt

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOORETYPES_H
