//===- Types.h - ESI type system -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_TYPES_H
#define ESI_TYPES_H

#include <algorithm>
#include <any>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "esi/Common.h"
#include "esi/Values.h" // For BitVector / Int / UInt

namespace esi {

/// Root class of the ESI type system.
class Type {
public:
  using ID = std::string;
  Type(const ID &id) : id(id) {}
  virtual ~Type() = default;

  ID getID() const { return id; }
  virtual std::ptrdiff_t getBitWidth() const { return -1; }

  /// Serialize an object to a MutableBitVector (LSB-first stream). The object
  /// should be passed via std::any. Implementations append fields in the order
  /// they are iterated (the first serialized field occupies the
  /// least-significant bits of the result).
  virtual MutableBitVector serialize(const std::any &obj) const {
    throw std::runtime_error("Serialization not implemented for type " + id);
  }

  /// Deserialize from a BitVector stream (LSB-first). Implementations consume
  /// bits from 'data' in-place (via logical right shifts) and return the
  /// reconstructed value. Remaining bits stay in 'data'.
  virtual std::any deserialize(BitVector &data) const {
    throw std::runtime_error("Deserialization not implemented for type " + id);
  }

  // Deserialize from a MessageData buffer. Maps the MessageData onto a
  // MutableBitVector, and proceeds with regular MutableBitVector
  // deserialization.
  std::any deserialize(const MessageData &data) const {
    auto bv = MutableBitVector(std::vector<uint8_t>(data.getData()));
    return deserialize(bv);
  }

  /// Ensure that a std::any object is valid for this type. Throws
  /// std::runtime_error if the object is not valid.
  virtual void ensureValid(const std::any &obj) const {
    throw std::runtime_error("Validation not implemented for type " + id);
  }

  // Check if a std::any object is valid for this type. Returns an optional
  // error message if the object is not valid, else, std::nullopt.
  std::optional<std::string> isValid(const std::any &obj) const {
    try {
      ensureValid(obj);
      return std::nullopt;
    } catch (const std::runtime_error &e) {
      return e.what();
    }
  }

  // Dump a textual representation of this type to the provided stream.
  void dump(std::ostream &os) const;

  // Return a textual representation of this type.
  std::string toString() const;

protected:
  ID id;
};

/// Bundles represent a collection of channels. Services exclusively expose
/// bundles (sometimes of just one channel). As such, they are the type of
/// accessible ports on an accelerator, from a host API perspective.
/// TODO: Add a good description of direction?
class BundleType : public Type {
public:
  enum Direction { To, From };

  using ChannelVector =
      std::vector<std::tuple<std::string, Direction, const Type *>>;

  BundleType(const ID &id, const ChannelVector &channels)
      : Type(id), channels(channels) {}

  const ChannelVector &getChannels() const { return channels; }
  std::ptrdiff_t getBitWidth() const override { return -1; };

  std::pair<const Type *, Direction> findChannel(std::string name) const;

protected:
  ChannelVector channels;
};

/// Channels are the basic communication primitives. They are unidirectional and
/// carry one values of one type.
class ChannelType : public Type {
public:
  using Type::deserialize;
  ChannelType(const ID &id, const Type *inner) : Type(id), inner(inner) {}
  const Type *getInner() const { return inner; }
  std::ptrdiff_t getBitWidth() const override { return inner->getBitWidth(); };

  void ensureValid(const std::any &obj) const override;
  MutableBitVector serialize(const std::any &obj) const override;
  std::any deserialize(BitVector &data) const override;

private:
  const Type *inner;
};

/// The "void" type is a special type which can be used to represent no type.
class VoidType : public Type {
public:
  using Type::deserialize;
  VoidType(const ID &id) : Type(id) {}
  // 'void' is 1 bit by convention.
  std::ptrdiff_t getBitWidth() const override { return 1; };

  void ensureValid(const std::any &obj) const override;
  MutableBitVector serialize(const std::any &obj) const override;
  std::any deserialize(BitVector &data) const override;
};

/// The "any" type is a special type which can be used to represent any type, as
/// identified by the type id. Said type id is guaranteed to be present in the
/// manifest. Importantly, the "any" type id over the wire may not be a string
/// as it is in software.
class AnyType : public Type {
public:
  AnyType(const ID &id) : Type(id) {}
  std::ptrdiff_t getBitWidth() const override { return -1; };
};

/// Bit vectors include signed, unsigned, and signless integers.
class BitVectorType : public Type {
public:
  BitVectorType(const ID &id, uint64_t width) : Type(id), width(width) {}

  uint64_t getWidth() const { return width; }
  std::ptrdiff_t getBitWidth() const override { return getWidth(); };

private:
  uint64_t width;
};

/// Bits are just an array of bits. They are not interpreted as a number but are
/// identified in the manifest as "signless" ints.
class BitsType : public BitVectorType {
public:
  using BitVectorType::BitVectorType;
  using Type::deserialize;

  void ensureValid(const std::any &obj) const override;
  MutableBitVector serialize(const std::any &obj) const override;
  std::any deserialize(BitVector &data) const override;
};

/// Integers are bit vectors which may be signed or unsigned and are interpreted
/// as numbers.
class IntegerType : public BitVectorType {
public:
  using BitVectorType::BitVectorType;
};

/// Signed integer.
class SIntType : public IntegerType {
public:
  using IntegerType::IntegerType;
  using Type::deserialize;

  void ensureValid(const std::any &obj) const override;
  MutableBitVector serialize(const std::any &obj) const override;
  std::any deserialize(BitVector &data) const override;
};

/// Unsigned integer.
class UIntType : public IntegerType {
public:
  using IntegerType::IntegerType;
  using Type::deserialize;

  void ensureValid(const std::any &obj) const override;
  MutableBitVector serialize(const std::any &obj) const override;
  std::any deserialize(BitVector &data) const override;
};

/// Structs are an ordered collection of fields, each with a name and a type.
class StructType : public Type {
public:
  using FieldVector = std::vector<std::pair<std::string, const Type *>>;
  using Type::deserialize;

  StructType(const ID &id, const FieldVector &fields, bool reverse = true)
      : Type(id), fields(fields), reverse(reverse) {}

  const FieldVector &getFields() const { return fields; }
  std::ptrdiff_t getBitWidth() const override {
    std::ptrdiff_t size = 0;
    for (auto [name, ty] : getFields()) {
      std::ptrdiff_t fieldSize = ty->getBitWidth();
      if (fieldSize < 0)
        return -1;
      size += fieldSize;
    }
    return size;
  }

  void ensureValid(const std::any &obj) const override;
  MutableBitVector serialize(const std::any &obj) const override;
  std::any deserialize(BitVector &data) const override;

  // Returns whether this struct type should be reversed when
  // serializing/deserializing.
  // By default, a truthy value here makes StructType's compatible with system
  // verilog, which has reversed struct field ordering, wrt. C/software struct
  // ordering.
  bool isReverse() const { return reverse; }

private:
  FieldVector fields;
  bool reverse;
};

/// Arrays have a compile time specified (static) size and an element type.
class ArrayType : public Type {
public:
  ArrayType(const ID &id, const Type *elementType, uint64_t size,
            bool reverse = true)
      : Type(id), elementType(elementType), size(size), reverse(reverse) {}
  using Type::deserialize;

  const Type *getElementType() const { return elementType; }
  uint64_t getSize() const { return size; }
  bool isReverse() const { return reverse; }
  std::ptrdiff_t getBitWidth() const override {
    std::ptrdiff_t elementSize = elementType->getBitWidth();
    if (elementSize < 0)
      return -1;
    return elementSize * size;
  }

  void ensureValid(const std::any &obj) const override;
  MutableBitVector serialize(const std::any &obj) const override;
  std::any deserialize(BitVector &data) const override;

private:
  const Type *elementType;
  uint64_t size;
  // 'reverse' controls whether array elements are reversed during
  // serialization/deserialization (to match SystemVerilog/Python ordering
  // expectations).
  bool reverse;
};

/// Windows represent a fixed-size sliding window over a stream of data.
/// They define an "into" type (the data structure being windowed) and a
/// "loweredType" (the hardware representation including control signals).
class WindowType : public Type {
public:
  /// Frame information describing which fields are included in a particular
  /// frame.
  struct Frame {
    std::string name;
    std::vector<std::string> fields;
  };

  WindowType(const ID &id, const std::string &name, const Type *intoType,
             const Type *loweredType, const std::vector<Frame> &frames)
      : Type(id), name(name), intoType(intoType), loweredType(loweredType),
        frames(frames) {}

  const std::string &getName() const { return name; }
  const Type *getIntoType() const { return intoType; }
  const Type *getLoweredType() const { return loweredType; }
  const std::vector<Frame> &getFrames() const { return frames; }

  std::ptrdiff_t getBitWidth() const override {
    return loweredType->getBitWidth();
  }

private:
  std::string name;
  const Type *intoType;
  const Type *loweredType;
  std::vector<Frame> frames;
};

/// Lists represent variable-length sequences of elements of a single type.
/// Unlike arrays which have a fixed size, lists can have any length.
class ListType : public Type {
public:
  ListType(const ID &id, const Type *elementType)
      : Type(id), elementType(elementType) {}

  const Type *getElementType() const { return elementType; }

  std::ptrdiff_t getBitWidth() const override { return -1; }

private:
  const Type *elementType;
};

} // namespace esi

#endif // ESI_TYPES_H
