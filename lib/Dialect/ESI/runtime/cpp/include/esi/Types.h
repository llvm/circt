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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace esi {

/// Root class of the ESI type system.
class Type {
public:
  using ID = std::string;
  Type(const ID &id) : id(id) {}
  virtual ~Type() = default;

  ID getID() const { return id; }
  virtual std::ptrdiff_t getBitWidth() const { return -1; }

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

  std::pair<const Type *, Direction> findChannel(std::string name) const {
    for (auto [channelName, dir, type] : channels)
      if (channelName == name)
        return std::make_pair(type, dir);
    throw std::runtime_error("Channel '" + name + "' not found in bundle");
  }

protected:
  ChannelVector channels;
};

/// Channels are the basic communication primitives. They are unidirectional and
/// carry one values of one type.
class ChannelType : public Type {
public:
  ChannelType(const ID &id, const Type *inner) : Type(id), inner(inner) {}
  const Type *getInner() const { return inner; }
  std::ptrdiff_t getBitWidth() const override { return inner->getBitWidth(); };

private:
  const Type *inner;
};

/// The "void" type is a special type which can be used to represent no type.
class VoidType : public Type {
public:
  VoidType(const ID &id) : Type(id) {}
  // 'void' is 1 bit by convention.
  std::ptrdiff_t getBitWidth() const override { return 1; };
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
};

/// Unsigned integer.
class UIntType : public IntegerType {
public:
  using IntegerType::IntegerType;
};

/// Structs are an ordered collection of fields, each with a name and a type.
class StructType : public Type {
public:
  using FieldVector = std::vector<std::pair<std::string, const Type *>>;

  StructType(const ID &id, const FieldVector &fields)
      : Type(id), fields(fields) {}

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

private:
  FieldVector fields;
};

/// Arrays have a compile time specified (static) size and an element type.
class ArrayType : public Type {
public:
  ArrayType(const ID &id, const Type *elementType, uint64_t size)
      : Type(id), elementType(elementType), size(size) {}

  const Type *getElementType() const { return elementType; }
  uint64_t getSize() const { return size; }
  std::ptrdiff_t getBitWidth() const override {
    std::ptrdiff_t elementSize = elementType->getBitWidth();
    if (elementSize < 0)
      return -1;
    return elementSize * size;
  }

private:
  const Type *elementType;
  uint64_t size;
};

} // namespace esi

#endif // ESI_TYPES_H
