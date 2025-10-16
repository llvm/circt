//===- Types.cpp - ESI type system -----------------------------*- C++ -*-===//
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

#include "esi/Types.h"
#include <algorithm>
#include <cstring>
#include <format>
#include <span>

namespace esi {

std::pair<const Type *, BundleType::Direction>
BundleType::findChannel(std::string name) const {
  for (auto [channelName, dir, type] : channels)
    if (channelName == name)
      return std::make_pair(type, dir);
  throw std::runtime_error(
      std::format("Channel '{}' not found in bundle", name));
}

void ChannelType::ensureValid(const std::any &obj) const {
  return inner->ensureValid(obj);
}

MessageData ChannelType::serialize(const std::any &obj) const {
  return inner->serialize(obj);
}

std::pair<std::any, std::span<const uint8_t>>
ChannelType::deserialize(std::span<const uint8_t> data) const {
  return inner->deserialize(data);
}

void VoidType::ensureValid(const std::any &obj) const {
  // Void type should be represented by an empty std::any or nullptr
  if (!obj.has_value())
    return;

  try {
    std::any_cast<std::nullptr_t>(obj);
    return;
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error(
        "void type must be represented by empty std::any or nullptr");
  }
}

MessageData VoidType::serialize(const std::any &obj) const {
  ensureValid(obj);

  // By convention, void is represented by a single byte of value 0
  std::vector<uint8_t> data = {0};
  return MessageData(std::move(data));
}

std::pair<std::any, std::span<const uint8_t>>
VoidType::deserialize(std::span<const uint8_t> data) const {
  if (data.size() == 0)
    throw std::runtime_error("void type cannot be represented by empty data");

  // Extract one byte and return the rest as a subspan
  return {std::any{}, data.subspan(1)};
}

void BitsType::ensureValid(const std::any &obj) const {
  try {
    auto data = std::any_cast<std::vector<uint8_t>>(obj);
    size_t expectedSize = (getWidth() + 7) / 8; // Round up to nearest byte
    if (data.size() != expectedSize) {
      throw std::runtime_error(std::format(
          "wrong size: expected {} bytes, got {}", expectedSize, data.size()));
    }
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error("must be std::vector<uint8_t>");
  }
}

MessageData BitsType::serialize(const std::any &obj) const {
  ensureValid(obj);

  auto data = std::any_cast<std::vector<uint8_t>>(obj);
  return MessageData(data);
}

std::pair<std::any, std::span<const std::uint8_t>>
BitsType::deserialize(std::span<const std::uint8_t> data) const {
  size_t size = (getWidth() + 7) / 8; // Round up to nearest byte
  if (data.size() < size)
    throw std::runtime_error("Insufficient data for bits type");

  // Create result vector from the span data
  std::vector<uint8_t> result(data.data(), data.data() + size);

  // Return remaining data as a subspan - zero copy!
  return {std::any(result), data.subspan(size)};
}

// Helper function to extract signed integer-like values from std::any
static int64_t getIntLikeFromAny(const std::any &obj) {
  const std::type_info &type = obj.type();

  if (type == typeid(int64_t))
    return std::any_cast<int64_t>(obj);
  if (type == typeid(int32_t))
    return static_cast<int64_t>(std::any_cast<int32_t>(obj));
  if (type == typeid(int16_t))
    return static_cast<int64_t>(std::any_cast<int16_t>(obj));
  if (type == typeid(int8_t))
    return static_cast<int64_t>(std::any_cast<int8_t>(obj));

  throw std::bad_any_cast();
}

// Helper function to extract unsigned integer-like values from std::any
static uint64_t getUIntLikeFromAny(const std::any &obj) {
  const std::type_info &type = obj.type();

  if (type == typeid(uint64_t))
    return std::any_cast<uint64_t>(obj);
  if (type == typeid(uint32_t))
    return static_cast<uint64_t>(std::any_cast<uint32_t>(obj));
  if (type == typeid(uint16_t))
    return static_cast<uint64_t>(std::any_cast<uint16_t>(obj));
  if (type == typeid(uint8_t))
    return static_cast<uint64_t>(std::any_cast<uint8_t>(obj));

  throw std::bad_any_cast();
}

void SIntType::ensureValid(const std::any &obj) const {
  if (getWidth() > 64)
    throw std::runtime_error("Width exceeds 64 bits");

  int64_t minVal = -(1LL << (getWidth() - 1));
  int64_t maxVal = (1LL << (getWidth() - 1)) - 1;
  if (getWidth() == 64) {
    // For 64-bit, we use the full range
    minVal = INT64_MIN;
    maxVal = INT64_MAX;
  }
  try {
    int64_t value = getIntLikeFromAny(obj);
    if (value < minVal || value > maxVal) {
      throw std::runtime_error(
          std::format("value {} out of range for {}-bit signed integer", value,
                      getWidth()));
    }
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error(
        "must be a signed integer type (int8_t, int16_t, int32_t, or int64_t)");
  }
}

MessageData SIntType::serialize(const std::any &obj) const {
  ensureValid(obj);

  int64_t value = getIntLikeFromAny(obj);
  size_t byteSize = (getWidth() + 7) / 8;

  // Use pointer casting for performance
  return MessageData(reinterpret_cast<const uint8_t *>(&value), byteSize);
}

std::pair<std::any, std::span<const std::uint8_t>>
SIntType::deserialize(std::span<const std::uint8_t> data) const {
  if (getWidth() > 64)
    throw std::runtime_error("Width exceeds 64 bits");

  size_t byteSize = (getWidth() + 7) / 8;
  if (data.size() < byteSize)
    throw std::runtime_error("Insufficient data for sint type");

  // Use pointer casting with sign extension
  uint64_t value = 0;
  std::memcpy(&value, data.data(), byteSize);

  // Sign extension
  if (getWidth() < 64 && (value & (1ULL << (getWidth() - 1)))) {
    uint64_t signExtension = ~((1ULL << getWidth()) - 1);
    value |= signExtension;
  }

  int64_t signedValue = static_cast<int64_t>(value);

  // Return the appropriate integer type based on bit width
  std::any result;
  if (getWidth() <= 8) {
    result = std::any(static_cast<int8_t>(signedValue));
  } else if (getWidth() <= 16) {
    result = std::any(static_cast<int16_t>(signedValue));
  } else if (getWidth() <= 32) {
    result = std::any(static_cast<int32_t>(signedValue));
  } else {
    result = std::any(signedValue);
  }

  // Return remaining data as a subspan - zero copy!
  return {result, data.subspan(byteSize)};
}

void UIntType::ensureValid(const std::any &obj) const {
  if (getWidth() > 64)
    throw std::runtime_error("Width exceeds 64 bits");

  uint64_t maxVal =
      (getWidth() == 64) ? UINT64_MAX : ((1ULL << getWidth()) - 1);
  try {
    uint64_t value = getUIntLikeFromAny(obj);
    if (value > maxVal) {
      throw std::runtime_error(
          std::format("value {} out of range for {}-bit unsigned integer",
                      value, getWidth()));
    }
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error("must be an unsigned integer type (uint8_t, "
                             "uint16_t, uint32_t, or uint64_t)");
  }
}

MessageData UIntType::serialize(const std::any &obj) const {
  ensureValid(obj);

  uint64_t value = getUIntLikeFromAny(obj);
  size_t byteSize = (getWidth() + 7) / 8;

  // Use pointer casting for performance
  return MessageData(reinterpret_cast<const uint8_t *>(&value), byteSize);
}

std::pair<std::any, std::span<const std::uint8_t>>
UIntType::deserialize(std::span<const std::uint8_t> data) const {
  if (getWidth() > 64)
    throw std::runtime_error("Width exceeds 64 bits");

  size_t byteSize = (getWidth() + 7) / 8;
  if (data.size() < byteSize)
    throw std::runtime_error("Insufficient data for uint type");

  // Use pointer casting for performance
  uint64_t value = 0;
  std::memcpy(&value, data.data(), byteSize);

  // Return the appropriate integer type based on bit width
  std::any result;
  if (getWidth() <= 8) {
    result = std::any(static_cast<uint8_t>(value));
  } else if (getWidth() <= 16) {
    result = std::any(static_cast<uint16_t>(value));
  } else if (getWidth() <= 32) {
    result = std::any(static_cast<uint32_t>(value));
  } else {
    result = std::any(value);
  }

  // Return remaining data as a subspan - zero copy!
  return {result, data.subspan(byteSize)};
}

void StructType::ensureValid(const std::any &obj) const {
  try {
    auto structData = std::any_cast<std::map<std::string, std::any>>(obj);

    if (structData.size() != fields.size()) {
      throw std::runtime_error(std::format("struct has {} fields, expected {}",
                                           structData.size(), fields.size()));
    }

    for (const auto &[fieldName, fieldType] : fields) {
      if (fieldType->getBitWidth() % 8 != 0)
        throw std::runtime_error(std::format(
            "C++ ser/de of struct types only supports "
            "structs with byte-aligned fields, but field '{}' has {} bits",
            fieldName, fieldType->getBitWidth()));

      auto it = structData.find(fieldName);
      if (it == structData.end())
        throw std::runtime_error(std::format("missing field '{}'", fieldName));

      try {
        fieldType->ensureValid(it->second);
      } catch (const std::runtime_error &e) {
        throw std::runtime_error(
            std::format("invalid field '{}': {}", fieldName, e.what()));
      }
    }
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error("must be std::map<std::string, std::any>");
  }
}

MessageData StructType::serialize(const std::any &obj) const {
  ensureValid(obj);

  auto structData = std::any_cast<std::map<std::string, std::any>>(obj);

  // Pre-allocate space for performance
  std::ptrdiff_t totalSize = getBitWidth();
  std::vector<uint8_t> result;
  if (totalSize > 0)
    result.reserve((totalSize + 7) / 8);

  auto serializeField = [&](const std::pair<std::string, const Type *> &field) {
    auto &fieldName = field.first;
    auto &fieldType = field.second;
    auto fieldData = fieldType->serialize(structData.at(fieldName));
    const auto &fieldBytes = fieldData.getData();
    result.insert(result.end(), fieldBytes.begin(), fieldBytes.end());
  };

  // Serialize fields in reverse order.
  if (isReverse()) {
    for (auto it = fields.rbegin(); it != fields.rend(); ++it)
      serializeField(*it);
  } else {
    for (const auto &field : fields)
      serializeField(field);
  }

  return MessageData(std::move(result));
}

std::pair<std::any, std::span<const std::uint8_t>>
StructType::deserialize(std::span<const std::uint8_t> data) const {
  std::map<std::string, std::any> result;
  std::span<const std::uint8_t> remaining = data;

  auto deserializeField =
      [&](const std::pair<std::string, const Type *> &field) {
        auto &fieldName = field.first;
        auto &fieldType = field.second;
        auto [fieldValue, newRemaining] = fieldType->deserialize(remaining);
        result[fieldName] = fieldValue;
        remaining = newRemaining;
      };

  if (isReverse()) {
    // Deserialize fields in reverse order.
    for (auto it = fields.rbegin(); it != fields.rend(); ++it)
      deserializeField(*it);
  } else {
    for (const auto &field : fields)
      deserializeField(field);
  }

  return {std::any(result), remaining};
}

void ArrayType::ensureValid(const std::any &obj) const {
  try {
    auto arrayData = std::any_cast<std::vector<std::any>>(obj);

    if (arrayData.size() != size) {
      throw std::runtime_error(std::format("array has {} elements, expected {}",
                                           arrayData.size(), size));
    }

    for (size_t i = 0; i < arrayData.size(); ++i) {
      try {
        elementType->ensureValid(arrayData[i]);
      } catch (const std::runtime_error &e) {
        throw std::runtime_error(
            std::format("invalid element {}: {}", i, e.what()));
      }
    }
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error("must be std::vector<std::any>");
  }
}

MessageData ArrayType::serialize(const std::any &obj) const {
  ensureValid(obj);

  auto arrayData = std::any_cast<std::vector<std::any>>(obj);

  // Pre-allocate space for performance
  std::ptrdiff_t totalSize = getBitWidth();
  std::vector<uint8_t> result;
  if (totalSize > 0)
    result.reserve((totalSize + 7) / 8);

  if (isReverse()) {
    for (auto it = arrayData.rbegin(); it != arrayData.rend(); ++it) {
      auto elementData = elementType->serialize(*it);
      const auto &elementBytes = elementData.getData();
      result.insert(result.end(), elementBytes.begin(), elementBytes.end());
    }
  } else {
    for (const auto &elem : arrayData) {
      auto elementData = elementType->serialize(elem);
      const auto &elementBytes = elementData.getData();
      result.insert(result.end(), elementBytes.begin(), elementBytes.end());
    }
  }

  return MessageData(std::move(result));
}

std::pair<std::any, std::span<const std::uint8_t>>
ArrayType::deserialize(std::span<const std::uint8_t> data) const {
  std::vector<std::any> result;
  result.reserve(size); // Pre-allocate for performance
  std::span<const std::uint8_t> remaining = data;
  for (uint64_t i = 0; i < size; ++i) {
    auto [elementValue, newRemaining] = elementType->deserialize(remaining);
    result.push_back(elementValue);
    remaining = newRemaining;
  }
  // If elements were serialized in reverse order, restore original ordering.
  if (isReverse())
    std::reverse(result.begin(), result.end());

  return {std::any(result), remaining};
}

} // namespace esi
