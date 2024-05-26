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
#include <variant>

namespace circt {
namespace moore {

/// The number of values each bit of a type can assume.
enum class Domain {
  /// Two-valued types such as `bit` or `int`.
  TwoValued,
  /// Four-valued types such as `logic` or `integer`.
  FourValued,
};

/// Whether a type is signed or unsigned.
enum class Sign {
  /// An `unsigned` type.
  Unsigned,
  /// A `signed` type.
  Signed,
};

/// Map a `Sign` to the corresponding keyword.
StringRef getKeywordFromSign(const Sign &sign);
/// Map the keywords `unsigned` and `signed` to the corresponding `Sign`.
std::optional<Sign> getSignFromKeyword(StringRef keyword);

template <typename Os>
Os &operator<<(Os &os, const Sign &sign) {
  os << getKeywordFromSign(sign);
  return os;
}

namespace detail {
struct StructTypeStorage;
} // namespace detail

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

class PackedType;
class StringType;
class ChandleType;
class EventType;
class RealType;
class UnpackedArrayType;
class OpenUnpackedArrayType;
class AssocArrayType;
class QueueType;
class UnpackedStructType;

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
                     QueueType, UnpackedStructType>(type);
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

class VoidType;
class IntType;
class ArrayType;
class OpenArrayType;
class PackedStructType;

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
    return llvm::isa<VoidType, IntType, ArrayType, OpenArrayType,
                     PackedStructType>(type);
  }

  /// Get the value domain of this type.
  Domain getDomain() const;

  /// Get the size of this type in bits.
  ///
  /// Returns `None` if any of the type's dimensions is unsized.
  std::optional<unsigned> getBitSize() const;

protected:
  using UnpackedType::UnpackedType;
};

//===----------------------------------------------------------------------===//
// Packed and Unpacked Structs
//===----------------------------------------------------------------------===//

/// Whether a struct is a `struct`, `union`, or `union tagged`.
enum class StructKind {
  /// A `struct`.
  Struct,
  /// A `union`.
  Union,
  /// A `union tagged`.
  TaggedUnion,
};

/// Map a `StructKind` to the corresponding mnemonic.
StringRef getMnemonicFromStructKind(StructKind kind);
/// Map a mnemonic to the corresponding `StructKind`.
std::optional<StructKind> getStructKindFromMnemonic(StringRef mnemonic);

template <typename Os>
Os &operator<<(Os &os, const StructKind &kind) {
  static constexpr StringRef keywords[] = {"struct", "union", "union tagged"};
  os << keywords[static_cast<unsigned>(kind)];
  return os;
}

/// A member of a struct.
struct StructMember {
  /// The name of this member.
  StringAttr name;
  /// The type of this member.
  UnpackedType type;

  bool operator==(const StructMember &other) const {
    return name == other.name && type == other.type;
  }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const StructMember &x) {
  return llvm::hash_combine(x.name, x.type);
}

/// A struct.
///
/// This represents both packed and unpacked structs. Which one it is depends on
/// whether this struct is embedded in a `PackedStructType` or a
/// `UnpackedStructType`. For the packed version the struct members are
/// guaranteed to be packed types as well.
struct Struct {
  /// Whether this is a `struct`, `union`, or `union tagged`.
  StructKind kind;
  /// The list of members.
  SmallVector<StructMember, 4> members;
  /// The value domain of this struct. If all members are two-valued, the
  /// overall struct is two-valued. Otherwise the struct is four-valued.
  Domain domain;
  /// The size of this struct in bits. This is `None` if any member type has an
  /// unknown size. This is commonly the case for unpacked member types, or
  /// dimensions with unknown size such as `[]` or `[$]`.
  std::optional<unsigned> bitSize;

  /// Create a new struct.
  Struct(StructKind kind, ArrayRef<StructMember> members);
};

/// A packed struct.
class PackedStructType : public Type::TypeBase<PackedStructType, PackedType,
                                               detail::StructTypeStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static PackedStructType get(MLIRContext *context, StructKind kind,
                              ArrayRef<StructMember> members);
  static PackedStructType get(MLIRContext *context, const Struct &strukt) {
    return get(context, strukt.kind, strukt.members);
  }

  /// Get the struct definition.
  const Struct &getStruct() const;

  /// Allow implicit casts from `PackedStructType` to the actual struct
  /// definition.
  operator const Struct &() const { return getStruct(); }

  static constexpr StringLiteral name = "moore.packed_struct";

protected:
  using Base::Base;
};

/// An unpacked struct.
class UnpackedStructType
    : public Type::TypeBase<UnpackedStructType, UnpackedType,
                            detail::StructTypeStorage,
                            ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedStructType get(MLIRContext *context, StructKind kind,
                                ArrayRef<StructMember> members);
  static UnpackedStructType get(MLIRContext *context, const Struct &strukt) {
    return get(context, strukt.kind, strukt.members);
  }

  /// Get the struct definition.
  const Struct &getStruct() const;

  /// Allow implicit casts from `UnpackedStructType` to the actual struct
  /// definition.
  operator const Struct &() const { return getStruct(); }

  static constexpr StringLiteral name = "moore.unpacked_struct";

protected:
  using Base::Base;
};

} // namespace moore
} // namespace circt

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOORETYPES_H
