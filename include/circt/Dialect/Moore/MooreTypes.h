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

/// Which side is greater in a range `[a:b]`.
enum class RangeDir {
  /// `a < b`
  Up,
  /// `a >= b`
  Down,
};

/// The `[a:b]` part in a vector/array type such as `logic [a:b]`.
struct Range {
  /// The total number of bits, given as `|a-b|+1`.
  unsigned size;
  /// The direction of the vector, i.e. whether `a > b` or `a < b`.
  RangeDir dir;
  /// The starting offset of the range.
  int offset;

  /// Construct a range `[size-1:0]`.
  explicit Range(unsigned size) : Range(size, RangeDir::Down, 0) {}

  /// Construct a range `[offset+size-1:offset]` if `dir` is `Down`, or
  /// `[offset:offset+size-1]` if `dir` is `Up`.
  Range(unsigned size, RangeDir dir, int offset)
      : size(size), dir(dir), offset(offset) {}

  /// Construct a range [left:right]`, with the direction inferred as `Down` if
  /// `left >= right`, or `Up` otherwise.
  Range(int left, int right) {
    if (left >= right) {
      size = left + 1 - right;
      dir = RangeDir::Down;
      offset = right;
    } else {
      size = right + 1 - left;
      dir = RangeDir::Up;
      offset = left;
    }
  }

  bool operator==(const Range &other) const {
    return size == other.size && dir == other.dir && offset == other.offset;
  }

  /// Get the `$left` dimension.
  int left() const { return dir == RangeDir::Up ? low() : high(); }
  /// Get the `$right` dimension.
  int right() const { return dir == RangeDir::Up ? high() : low(); }
  /// Get the `$low` dimension.
  int low() const { return offset; }
  /// Get the `$high` dimension.
  int high() const { return offset + size - 1; }
  /// Get the `$increment` size.
  int increment() const { return dir == RangeDir::Up ? 1 : -1; }

  /// Format this range as a string.
  std::string toString() const {
    std::string buffer;
    llvm::raw_string_ostream(buffer) << *this;
    return buffer;
  }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const Range &x) {
  return llvm::hash_combine(x.size, x.dir, x.offset);
}

template <typename Os>
Os &operator<<(Os &os, const Range &range) {
  os << range.left() << ":" << range.right();
  return os;
}

class PackedType;

namespace detail {
struct DimStorage;
struct UnsizedDimStorage;
struct RangeDimStorage;
struct SizedDimStorage;
struct AssocDimStorage;
struct StructTypeStorage;
} // namespace detail

/// Base class for all SystemVerilog types in the Moore dialect.
class SVType : public Type {
protected:
  using Type::Type;
};

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

class PackedType;
class StringType;
class ChandleType;
class EventType;
class RealType;
class UnpackedDim;
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
class UnpackedType : public SVType {
public:
  static bool classof(Type type) {
    return llvm::isa<PackedType>(type) || llvm::isa<StringType>(type) ||
           llvm::isa<ChandleType>(type) || llvm::isa<EventType>(type) ||
           llvm::isa<RealType>(type) || llvm::isa<UnpackedDim>(type) ||
           llvm::isa<UnpackedStructType>(type);
  }

  /// Get the value domain of this type.
  Domain getDomain() const;

  /// Get the size of this type in bits.
  ///
  /// Returns `None` if any of the type's dimensions is unsized, associative, or
  /// a queue, or the core type itself has no known size.
  std::optional<unsigned> getBitSize() const;

protected:
  using SVType::SVType;
};

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

class VoidType;
class IntType;
class PackedDim;
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
    return llvm::isa<VoidType>(type) || llvm::isa<IntType>(type) ||
           llvm::isa<PackedDim>(type) || llvm::isa<PackedStructType>(type);
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
// Packed Dimensions
//===----------------------------------------------------------------------===//

class PackedRangeDim;
class PackedUnsizedDim;

/// A packed dimension.
class PackedDim : public PackedType {
public:
  static bool classof(Type type) {
    return llvm::isa<PackedRangeDim>(type) || llvm::isa<PackedUnsizedDim>(type);
  }

  /// Get the element type of the dimension. This is the `x` in `x[a:b]`.
  PackedType getInner() const;

  /// Get the dimension's range, or `None` if it is unsized.
  std::optional<Range> getRange() const;
  /// Get the dimension's size, or `None` if it is unsized.
  std::optional<unsigned> getSize() const;

protected:
  using PackedType::PackedType;
  const detail::DimStorage *getImpl() const;
};

/// A packed unsized dimension, like `[]`.
class PackedUnsizedDim : public Type::TypeBase<PackedUnsizedDim, PackedDim,
                                               detail::UnsizedDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static PackedUnsizedDim get(PackedType inner);

  static constexpr StringLiteral name = "moore.packed_unsized_dim";

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// A packed range dimension, like `[a:b]`.
class PackedRangeDim
    : public Type::TypeBase<PackedRangeDim, PackedDim, detail::RangeDimStorage,
                            ::mlir::TypeTrait::IsMutable> {
public:
  static PackedRangeDim get(PackedType inner, Range range);

  /// Get a packed range with arguments forwarded to the `Range` constructor.
  /// See `Range::Range` for details.
  template <typename... Args>
  static PackedRangeDim get(PackedType inner, Args... args) {
    return get(inner, Range(args...));
  }

  /// Get the range of this dimension.
  Range getRange() const;

  /// Allow implicit casts from `PackedRangeDim` to the actual range.
  operator Range() const { return getRange(); }

  static constexpr StringLiteral name = "moore.packed_range_dim";

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

//===----------------------------------------------------------------------===//
// Unpacked Dimensions
//===----------------------------------------------------------------------===//

class UnpackedUnsizedDim;
class UnpackedArrayDim;
class UnpackedRangeDim;
class UnpackedAssocDim;
class UnpackedQueueDim;

/// An unpacked dimension.
class UnpackedDim : public UnpackedType {
public:
  static bool classof(Type type) {
    return llvm::isa<UnpackedUnsizedDim>(type) ||
           llvm::isa<UnpackedArrayDim>(type) ||
           llvm::isa<UnpackedRangeDim>(type) ||
           llvm::isa<UnpackedAssocDim>(type) ||
           llvm::isa<UnpackedQueueDim>(type);
  }

  /// Get the element type of the dimension. This is the `x` in `x[a:b]`.
  UnpackedType getInner() const;

protected:
  using UnpackedType::UnpackedType;
  const detail::DimStorage *getImpl() const;
};

/// An unpacked unsized dimension, like `[]`.
class UnpackedUnsizedDim
    : public Type::TypeBase<UnpackedUnsizedDim, UnpackedDim,
                            detail::UnsizedDimStorage,
                            ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedUnsizedDim get(UnpackedType inner);

  static constexpr StringLiteral name = "moore.unpacked_unsized_dim";

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked array dimension, like `[a]`.
class UnpackedArrayDim : public Type::TypeBase<UnpackedArrayDim, UnpackedDim,
                                               detail::SizedDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedArrayDim get(UnpackedType inner, unsigned size);

  /// Get the size of the array, i.e. the `a` in `[a]`.
  unsigned getSize() const;

  static constexpr StringLiteral name = "moore.unpacked_array_dim";

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked range dimension, like `[a:b]`.
class UnpackedRangeDim : public Type::TypeBase<UnpackedRangeDim, UnpackedDim,
                                               detail::RangeDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedRangeDim get(UnpackedType inner, Range range);

  /// Get a packed range with arguments forwarded to the `Range` constructor.
  /// See `Range::Range` for details.
  template <typename... Args>
  static UnpackedRangeDim get(UnpackedType inner, Args... args) {
    return get(inner, Range(args...));
  }

  /// Get the range of this dimension.
  Range getRange() const;

  /// Allow implicit casts from `UnpackedRangeDim` to the actual range.
  operator Range() const { return getRange(); }

  static constexpr StringLiteral name = "moore.unpacked_range_dim";

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked associative dimension, like `[T]` or `[*]`.
///
/// Associative arrays in SystemVerilog can have a concrete index type (`[T]`),
/// or a wildcard index type (`[*]`, §7.8.1). The latter is exceptionally
/// strange, as it applies only to integer indices, but supports arbitrarily
/// sized indices by always removing leading zeros from any index that is used
/// in the lookup. This is interesting if a `string` is used to index into such
/// an array, because strings are automatically cast to a bit vector of
/// equivalent size, which results in a sort-of string key lookup. However, note
/// that there are also dedicated semantics for using `string` as the actual
/// index type (§7.8.2).
///
/// See IEEE 1800-2017 §7.8 "Associative arrays".
class UnpackedAssocDim : public Type::TypeBase<UnpackedAssocDim, UnpackedDim,
                                               detail::AssocDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedAssocDim get(UnpackedType inner, UnpackedType indexType = {});

  /// Get the index type of the associative dimension. This returns either the
  /// type `T` in a dimension `[T]`, or a null type in a dimension `[*]`.
  UnpackedType getIndexType() const;

  static constexpr StringLiteral name = "moore.unpacked_assoc_dim";

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked queue dimension with optional bound, like `[$]` or `[$:a]`.
class UnpackedQueueDim : public Type::TypeBase<UnpackedQueueDim, UnpackedDim,
                                               detail::SizedDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedQueueDim get(UnpackedType inner,
                              std::optional<unsigned> bound = {});

  /// Get the bound of the queue, i.e. the `a` in `[$:a]`. Returns `None` if the
  /// queue is unbounded.
  std::optional<unsigned> getBound() const;

  static constexpr StringLiteral name = "moore.unpacked_queue_dim";

protected:
  using Base::Base;
  friend struct detail::DimStorage;
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

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct DenseMapInfo<circt::moore::Range> {
  using Range = circt::moore::Range;
  static inline Range getEmptyKey() { return Range(-1); }
  static inline Range getTombstoneKey() { return Range(-2); }
  static unsigned getHashValue(const Range &x) {
    return circt::moore::hash_value(x);
  }
  static bool isEqual(const Range &lhs, const Range &rhs) { return lhs == rhs; }
};

} // namespace llvm

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOORETYPES_H
