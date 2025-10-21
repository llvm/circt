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
#include "esi/Values.h"
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

MutableBitVector ChannelType::serialize(const std::any &obj) const {
  return inner->serialize(obj);
}

std::any ChannelType::deserialize(MutableBitVector &data) const {
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

std::any VoidType::deserialize(MutableBitVector &data) const {
  if (data.width() < 8)
    throw std::runtime_error("void type cannot be represented by empty data");
  // Extract one byte and return the rest. Check that the byte is 0.
  BitVector value = data.lsb(8);
  // Manually check if any bit is set (instead of using std::ranges::any_of)
  for (size_t i = 0; i < 8; ++i) {
    if (value.getBit(i))
      throw std::runtime_error("void type byte must be 0");
  }

  data >>= 8; // consume one byte (value ignored)
  return std::any{};
}

MutableBitVector VoidType::serialize(const std::any &obj) const {
  ensureValid(obj);
  // By convention, void is represented by a single byte of value 0.
  MutableBitVector bv(8);
  return bv;
}

void BitsType::ensureValid(const std::any &obj) const {
  try {
    auto data = std::any_cast<std::vector<uint8_t>>(obj);
    size_t expectedSize = (getWidth() + 7) / 8; // Round up to nearest byte
    if (data.size() != expectedSize) {
      throw std::runtime_error("wrong size: expected " +
                               std::to_string(expectedSize) + " bytes, got " +
                               std::to_string(data.size()));
    }
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error("must be std::vector<uint8_t>");
  }
}

MutableBitVector BitsType::serialize(const std::any &obj) const {
  ensureValid(obj);
  auto bytes = std::any_cast<std::vector<uint8_t>>(obj); // copy
  return MutableBitVector(std::move(bytes), getWidth());
}

std::any BitsType::deserialize(MutableBitVector &data) const {
  uint64_t w = getWidth();
  if (data.width() < w)
    throw std::runtime_error(std::format("Insufficient data for bits type. "
                                         " Expected {} bits, got {} bits",
                                         w, data.width()));
  BitVector view = data.slice(0, w);
  // Materialize into byte vector sized to width with fast path if aligned.
  size_t byteCount = (w + 7) / 8;
  std::vector<uint8_t> out(byteCount, 0);
  bool fastPath = false;
  if (w > 0) {
    try {
      auto span = view.getSpan(); // requires bitIndex==0
      fastPath = true;
      std::memcpy(out.data(), span.data(), byteCount);
      if (w % 8 != 0) {
        uint8_t mask = static_cast<uint8_t>((1u << (w % 8)) - 1u);
        out[byteCount - 1] &= mask;
      }
    } catch (const std::runtime_error &) {
      fastPath = false; // Fallback below
    }
  }
  if (!fastPath) {
    for (uint64_t i = 0; i < w; ++i)
      if (view.getBit(i))
        out[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
  }
  data >>= w;
  return std::any(out);
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

MutableBitVector SIntType::serialize(const std::any &obj) const {
  // Expect esi::Int of correct width.
  try {
    const Int &ival = std::any_cast<const Int &>(obj);
    if (static_cast<uint64_t>(ival.width()) != getWidth())
      throw std::runtime_error("Int width mismatch for SIntType serialize");
    // Copy bits into new MutableBitVector (owning) of this width.
    MutableBitVector out(getWidth());
    for (uint64_t i = 0; i < getWidth(); ++i)
      if (ival.getBit(i))
        out.setBit(i, true);
    return out;
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error("SIntType expects esi::Int for serialization");
  }
}

std::any SIntType::deserialize(MutableBitVector &data) const {
  uint64_t w = getWidth();
  if (data.width() < w)
    throw std::runtime_error("Insufficient data for sint type");
  Int val(data.slice(0, w));
  data >>= w;
  return std::any(val);
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

MutableBitVector UIntType::serialize(const std::any &obj) const {
  try {
    const UInt &uval = std::any_cast<const UInt &>(obj);
    if (static_cast<uint64_t>(uval.width()) != getWidth())
      throw std::runtime_error("UInt width mismatch for UIntType serialize");
    MutableBitVector out(getWidth());
    for (uint64_t i = 0; i < getWidth(); ++i)
      if (uval.getBit(i))
        out.setBit(i, true);
    return out;
  } catch (const std::bad_any_cast &) {
    throw std::runtime_error("UIntType expects esi::UInt for serialization");
  }
}

std::any UIntType::deserialize(MutableBitVector &data) const {
  uint64_t w = getWidth();
  if (data.width() < w)
    throw std::runtime_error("Insufficient data for uint type");
  UInt val(data.slice(0, w));
  data >>= w;
  return std::any(val);
}

void StructType::ensureValid(const std::any &obj) const {
  try {
    auto structData = std::any_cast<std::map<std::string, std::any>>(obj);

    if (structData.size() != fields.size()) {
      throw std::runtime_error(std::format("struct has {} fields, expected {}",
                                           structData.size(), fields.size()));
    }

    for (const auto &[fieldName, fieldType] : fields) {
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

MutableBitVector StructType::serialize(const std::any &obj) const {
  ensureValid(obj);
  auto structData = std::any_cast<std::map<std::string, std::any>>(obj);
  std::vector<BitVector> parts;
  parts.reserve(fields.size());
  uint64_t totalWidth = 0;
  auto handleField = [&](const std::pair<std::string, const Type *> &f) {
    const auto &name = f.first;
    const Type *ty = f.second;
    BitVector bv = ty->serialize(structData.at(name));
    totalWidth += bv.width();
    parts.push_back(std::move(bv));
  };
  if (isReverse()) {
    for (auto it = fields.rbegin(); it != fields.rend(); ++it)
      handleField(*it);
  } else {
    for (const auto &f : fields)
      handleField(f);
  }
  MutableBitVector out(totalWidth);
  uint64_t offset = 0;
  for (const auto &p : parts) {
    for (uint64_t i = 0; i < p.width(); ++i)
      if (p.getBit(i))
        out.setBit(offset + i, true);
    offset += p.width();
  }
  return out;
}

std::any StructType::deserialize(MutableBitVector &data) const {
  std::map<std::string, std::any> result;
  auto consumeField = [&](const std::pair<std::string, const Type *> &f) {
    const auto &name = f.first;
    const Type *ty = f.second;
    std::any value = ty->deserialize(data);
    result[name] = value;
  };
  if (isReverse()) {
    for (auto it = fields.rbegin(); it != fields.rend(); ++it)
      consumeField(*it);
  } else {
    for (const auto &f : fields)
      consumeField(f);
  }
  return std::any(result);
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

MutableBitVector ArrayType::serialize(const std::any &obj) const {
  ensureValid(obj);
  auto arrayData = std::any_cast<std::vector<std::any>>(obj);
  std::vector<BitVector> parts;
  parts.reserve(arrayData.size());
  uint64_t totalWidth = 0;
  auto pushElem = [&](const std::any &elem) {
    BitVector bv = elementType->serialize(elem);
    totalWidth += bv.width();
    parts.push_back(std::move(bv));
  };
  if (isReverse()) {
    for (auto it = arrayData.rbegin(); it != arrayData.rend(); ++it)
      pushElem(*it);
  } else {
    for (const auto &e : arrayData)
      pushElem(e);
  }
  MutableBitVector out(totalWidth);
  uint64_t offset = 0;
  for (const auto &p : parts) {
    for (uint64_t i = 0; i < p.width(); ++i)
      if (p.getBit(i))
        out.setBit(offset + i, true);
    offset += p.width();
  }
  return out;
}

std::any ArrayType::deserialize(MutableBitVector &data) const {
  std::vector<std::any> result;
  result.reserve(size);
  for (uint64_t i = 0; i < size; ++i) {
    std::any value = elementType->deserialize(data);
    result.push_back(value);
  }
  if (isReverse())
    std::reverse(result.begin(), result.end());
  return std::any(result);
}

} // namespace esi
