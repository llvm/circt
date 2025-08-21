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
#include <stdexcept>
#include <string>
#include <vector>

#include "esi/Common.h"

namespace esi {

/// Root class of the ESI type system.
class Type {
public:
  using ID = std::string;
  Type(const ID &id) : id(id) {}
  virtual ~Type() = default;

  ID getID() const { return id; }
  virtual std::ptrdiff_t getBitWidth() const { return -1; }

  /// Serialize an object to MessageData. The object should be passed as a
  /// std::any to provide type erasure. Returns a MessageData containing the
  /// serialized representation.
  virtual MessageData serialize(const std::any &obj) const {
    throw std::runtime_error("Serialization not implemented for type " + id);
  }

  /// Deserialize MessageData to an object. Returns the deserialized object
  /// as a std::any and the remaining MessageData bytes.
  virtual std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const {
    throw std::runtime_error("Deserialization not implemented for type " + id);
  }

  /// Check if a std::any object is valid for this type. Returns true if valid,
  /// false otherwise with an optional error message.
  virtual std::pair<bool, std::string> isValid(const std::any &obj) const {
    return {false, "Validation not implemented for type " + id};
  }

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

  std::pair<bool, std::string> isValid(const std::any &obj) const override {
    return inner->isValid(obj);
  }

  MessageData serialize(const std::any &obj) const override {
    return inner->serialize(obj);
  }

  std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const override {
    return inner->deserialize(data);
  }

private:
  const Type *inner;
};

/// The "void" type is a special type which can be used to represent no type.
class VoidType : public Type {
public:
  VoidType(const ID &id) : Type(id) {}
  // 'void' is 1 bit by convention.
  std::ptrdiff_t getBitWidth() const override { return 1; };

  std::pair<bool, std::string> isValid(const std::any &obj) const override {
    // Void type should be represented by an empty std::any or nullptr
    if (!obj.has_value()) {
      return {true, ""};
    }
    try {
      std::any_cast<std::nullptr_t>(obj);
      return {true, ""};
    } catch (const std::bad_any_cast &) {
      return {false,
              "void type must be represented by empty std::any or nullptr"};
    }
  }

  MessageData serialize(const std::any &obj) const override {
    auto [valid, reason] = isValid(obj);
    if (!valid) {
      throw std::runtime_error("Invalid object for void type: " + reason);
    }
    // By convention, void is represented by a single byte of value 0
    std::vector<uint8_t> data = {0};
    return MessageData(std::move(data));
  }

  std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const override {
    if (data.getSize() == 0) {
      throw std::runtime_error("void type cannot be represented by empty data");
    }
    // Extract one byte and return the rest
    std::vector<uint8_t> remaining(data.getData().begin() + 1,
                                   data.getData().end());
    return {std::any{}, MessageData(std::move(remaining))};
  }
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

  std::pair<bool, std::string> isValid(const std::any &obj) const override {
    try {
      auto data = std::any_cast<std::vector<uint8_t>>(obj);
      size_t expectedSize = (getWidth() + 7) / 8; // Round up to nearest byte
      if (data.size() != expectedSize) {
        return {false, "wrong size: expected " + std::to_string(expectedSize) +
                           " bytes, got " + std::to_string(data.size())};
      }
      return {true, ""};
    } catch (const std::bad_any_cast &) {
      return {false, "must be std::vector<uint8_t>"};
    }
  }

  MessageData serialize(const std::any &obj) const override {
    auto [valid, reason] = isValid(obj);
    if (!valid) {
      throw std::runtime_error("Invalid object for bits type: " + reason);
    }
    auto data = std::any_cast<std::vector<uint8_t>>(obj);
    return MessageData(data);
  }

  std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const override {
    size_t size = (getWidth() + 7) / 8; // Round up to nearest byte
    if (data.getSize() < size) {
      throw std::runtime_error("Insufficient data for bits type");
    }
    std::vector<uint8_t> result(data.getData().begin(),
                                data.getData().begin() + size);
    std::vector<uint8_t> remaining(data.getData().begin() + size,
                                   data.getData().end());
    return {std::any(result), MessageData(std::move(remaining))};
  }
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

  std::pair<bool, std::string> isValid(const std::any &obj) const override {
    if (getWidth() > 64)
      throw std::runtime_error("Width exceeds 64 bits");

    try {
      auto value = std::any_cast<int64_t>(obj);
      int64_t minVal = -(1LL << (getWidth() - 1));
      int64_t maxVal = (1LL << (getWidth() - 1)) - 1;
      if (getWidth() == 64) {
        // For 64-bit, we use the full range
        minVal = INT64_MIN;
        maxVal = INT64_MAX;
      }
      if (value < minVal || value > maxVal) {
        return {false, "value " + std::to_string(value) + " out of range for " +
                           std::to_string(getWidth()) + "-bit signed integer"};
      }
      return {true, ""};
    } catch (const std::bad_any_cast &) {
      return {false, "must be int64_t"};
    }
  }

  MessageData serialize(const std::any &obj) const override {
    auto [valid, reason] = isValid(obj);
    if (!valid) {
      throw std::runtime_error("Invalid object for sint type: " + reason);
    }
    auto value = std::any_cast<int64_t>(obj);
    size_t byteSize = (getWidth() + 7) / 8;
    std::vector<uint8_t> data(byteSize);

    // Little-endian serialization with sign extension
    uint64_t unsignedValue = static_cast<uint64_t>(value);
    for (size_t i = 0; i < byteSize; ++i) {
      data[i] = static_cast<uint8_t>((unsignedValue >> (i * 8)) & 0xFF);
    }
    return MessageData(std::move(data));
  }

  std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const override {
    if (getWidth() > 64)
      throw std::runtime_error("Width exceeds 64 bits");

    size_t byteSize = (getWidth() + 7) / 8;
    if (data.getSize() < byteSize) {
      throw std::runtime_error("Insufficient data for sint type");
    }

    uint64_t value = 0;
    // Little-endian deserialization
    for (size_t i = 0; i < byteSize; ++i) {
      value |= static_cast<uint64_t>(data.getData()[i]) << (i * 8);
    }

    // Sign extension
    if (getWidth() < 64 && (value & (1ULL << (getWidth() - 1)))) {
      // If the sign bit is set, extend it
      uint64_t signExtension = ~((1ULL << getWidth()) - 1);
      value |= signExtension;
    }

    int64_t signedValue = static_cast<int64_t>(value);
    std::vector<uint8_t> remaining(data.getData().begin() + byteSize,
                                   data.getData().end());
    return {std::any(signedValue), MessageData(std::move(remaining))};
  }
};

/// Unsigned integer.
class UIntType : public IntegerType {
public:
  using IntegerType::IntegerType;

  std::pair<bool, std::string> isValid(const std::any &obj) const override {
    if (getWidth() > 64)
      throw std::runtime_error("Width exceeds 64 bits");

    try {
      auto value = std::any_cast<uint64_t>(obj);
      uint64_t maxVal =
          (getWidth() == 64) ? UINT64_MAX : ((1ULL << getWidth()) - 1);
      if (value > maxVal) {
        return {false, "value " + std::to_string(value) + " out of range for " +
                           std::to_string(getWidth()) +
                           "-bit unsigned integer"};
      }
      return {true, ""};
    } catch (const std::bad_any_cast &) {
      return {false, "must be uint64_t"};
    }
  }

  MessageData serialize(const std::any &obj) const override {
    auto [valid, reason] = isValid(obj);
    if (!valid) {
      throw std::runtime_error("Invalid object for uint type: " + reason);
    }
    auto value = std::any_cast<uint64_t>(obj);
    size_t byteSize = (getWidth() + 7) / 8;
    std::vector<uint8_t> data(byteSize);

    // Little-endian serialization
    for (size_t i = 0; i < byteSize; ++i) {
      data[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
    }
    return MessageData(std::move(data));
  }

  std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const override {
    if (getWidth() > 64)
      throw std::runtime_error("Width exceeds 64 bits");

    size_t byteSize = (getWidth() + 7) / 8;
    if (data.getSize() < byteSize) {
      throw std::runtime_error("Insufficient data for uint type");
    }

    uint64_t value = 0;
    // Little-endian deserialization
    for (size_t i = 0; i < byteSize; ++i) {
      value |= static_cast<uint64_t>(data.getData()[i]) << (i * 8);
    }

    std::vector<uint8_t> remaining(data.getData().begin() + byteSize,
                                   data.getData().end());
    return {std::any(value), MessageData(std::move(remaining))};
  }
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

  std::pair<bool, std::string> isValid(const std::any &obj) const override {
    try {
      auto structData = std::any_cast<std::map<std::string, std::any>>(obj);

      if (structData.size() != fields.size()) {
        return {false, "struct has " + std::to_string(structData.size()) +
                           " fields, expected " +
                           std::to_string(fields.size())};
      }

      for (const auto &[fieldName, fieldType] : fields) {
        auto it = structData.find(fieldName);
        if (it == structData.end()) {
          return {false, "missing field '" + fieldName + "'"};
        }

        auto [fieldValid, fieldReason] = fieldType->isValid(it->second);
        if (!fieldValid) {
          return {false, "invalid field '" + fieldName + "': " + fieldReason};
        }
      }
      return {true, ""};
    } catch (const std::bad_any_cast &) {
      return {false, "must be std::map<std::string, std::any>"};
    }
  }

  MessageData serialize(const std::any &obj) const override {
    auto [valid, reason] = isValid(obj);
    if (!valid) {
      throw std::runtime_error("Invalid object for struct type: " + reason);
    }

    auto structData = std::any_cast<std::map<std::string, std::any>>(obj);
    std::vector<uint8_t> result;

    // Serialize fields in reverse order (to match Python implementation)
    for (auto it = fields.rbegin(); it != fields.rend(); ++it) {
      const auto &[fieldName, fieldType] = *it;
      auto fieldData = fieldType->serialize(structData.at(fieldName));
      const auto &fieldBytes = fieldData.getData();
      result.insert(result.end(), fieldBytes.begin(), fieldBytes.end());
    }

    return MessageData(std::move(result));
  }

  std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const override {
    std::map<std::string, std::any> result;
    MessageData remaining = data;

    // Deserialize fields in reverse order (to match Python implementation)
    for (auto it = fields.rbegin(); it != fields.rend(); ++it) {
      const auto &[fieldName, fieldType] = *it;
      auto [fieldValue, newRemaining] = fieldType->deserialize(remaining);
      result[fieldName] = fieldValue;
      remaining = std::move(newRemaining);
    }

    return {std::any(result), std::move(remaining)};
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

  std::pair<bool, std::string> isValid(const std::any &obj) const override {
    try {
      auto arrayData = std::any_cast<std::vector<std::any>>(obj);

      if (arrayData.size() != size) {
        return {false, "array has " + std::to_string(arrayData.size()) +
                           " elements, expected " + std::to_string(size)};
      }

      for (size_t i = 0; i < arrayData.size(); ++i) {
        auto [elementValid, elementReason] = elementType->isValid(arrayData[i]);
        if (!elementValid) {
          return {false, "invalid element " + std::to_string(i) + ": " +
                             elementReason};
        }
      }
      return {true, ""};
    } catch (const std::bad_any_cast &) {
      return {false, "must be std::vector<std::any>"};
    }
  }

  MessageData serialize(const std::any &obj) const override {
    auto [valid, reason] = isValid(obj);
    if (!valid) {
      throw std::runtime_error("Invalid object for array type: " + reason);
    }

    auto arrayData = std::any_cast<std::vector<std::any>>(obj);
    std::vector<uint8_t> result;

    // Serialize elements in reverse order (to match Python implementation)
    for (auto it = arrayData.rbegin(); it != arrayData.rend(); ++it) {
      auto elementData = elementType->serialize(*it);
      const auto &elementBytes = elementData.getData();
      result.insert(result.end(), elementBytes.begin(), elementBytes.end());
    }

    return MessageData(std::move(result));
  }

  std::pair<std::any, MessageData>
  deserialize(const MessageData &data) const override {
    std::vector<std::any> result;
    MessageData remaining = data;

    // Deserialize elements (will be in reverse order, so we'll reverse at the
    // end)
    for (uint64_t i = 0; i < size; ++i) {
      auto [elementValue, newRemaining] = elementType->deserialize(remaining);
      result.push_back(elementValue);
      remaining = std::move(newRemaining);
    }
    std::reverse(result.begin(), result.end());

    return {std::any(result), std::move(remaining)};
  }

private:
  const Type *elementType;
  uint64_t size;
};

} // namespace esi

#endif // ESI_TYPES_H
