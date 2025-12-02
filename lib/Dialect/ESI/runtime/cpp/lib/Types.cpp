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
#include <sstream>

namespace esi {

// NOLINTNEXTLINE(misc-no-recursion)
static void dumpType(std::ostream &os, const esi::Type *type, int level = 0) {
  if (auto *uintType = dynamic_cast<const esi::UIntType *>(type)) {
    os << "uint" << uintType->getBitWidth();
  } else if (auto *sintType = dynamic_cast<const esi::SIntType *>(type)) {
    os << "sint" << sintType->getBitWidth();
  } else if (auto *bitsType = dynamic_cast<const esi::BitsType *>(type)) {
    os << "bits" << bitsType->getBitWidth();
  } else if (dynamic_cast<const esi::VoidType *>(type)) {
    os << "void";
  } else if (dynamic_cast<const esi::AnyType *>(type)) {
    os << "any";
  } else if (auto *structType = dynamic_cast<const esi::StructType *>(type)) {
    os << "struct {" << std::endl;
    for (const auto &[name, fieldType] : structType->getFields()) {
      os << std::string(level + 2, ' ') << name << ": ";
      dumpType(os, fieldType, level + 1);
      os << "," << std::endl;
    }
    os << std::string(level, ' ') << "}";
  } else if (auto *arrayType = dynamic_cast<const esi::ArrayType *>(type)) {
    dumpType(os, arrayType->getElementType(), level + 1);
    os << "[" << arrayType->getSize() << "]";
  } else if (auto *channelType = dynamic_cast<const esi::ChannelType *>(type)) {
    os << "chan<";
    dumpType(os, channelType->getInner(), level + 1);
    os << ">";
  } else if (auto *bundleType = dynamic_cast<const esi::BundleType *>(type)) {
    os << "bundle {" << std::endl;
    for (const auto &[name, direction, fieldType] : bundleType->getChannels()) {
      os << std::string(level + 2, ' ')
         << std::format(
                "{} [{}]: ", direction == BundleType::To ? "to" : "from", name);
      dumpType(os, fieldType, level + 1);
      os << "," << std::endl;
    }
    os << std::string(level, ' ') << "}";
  } else if (auto *windowType = dynamic_cast<const esi::WindowType *>(type)) {
    os << "window[";
    dumpType(os, windowType->getIntoType(), level + 1);
    os << "]";
  } else if (auto *listType = dynamic_cast<const esi::ListType *>(type)) {
    os << "list<";
    dumpType(os, listType->getElementType(), level + 1);
    os << ">";
  } else {
    // For unknown types, just print the type ID
    os << type->getID();
  }
}

void Type::dump(std::ostream &os) const { dumpType(os, this); }

// Recurse through an ESI type and print an elaborated version of it.
std::string Type::toString() const {
  std::stringstream ss;
  dump(ss);
  return ss.str();
}

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

std::any ChannelType::deserialize(BitVector &data) const {
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

std::any VoidType::deserialize(BitVector &data) const {
  if (data.width() < 8)
    throw std::runtime_error("void type cannot be represented by empty data");
  // Extract one byte and return the rest. Check that the byte is 0.
  BitVector value = data.lsb(8);
  if (std::ranges::any_of(value, [](auto b) { return b; }))
    throw std::runtime_error("void type byte must be 0");

  data >>= 8;
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

std::any BitsType::deserialize(BitVector &data) const {
  uint64_t w = getWidth();
  if (data.width() < w)
    throw std::runtime_error(std::format("Insufficient data for bits type. "
                                         " Expected {} bits, got {} bits",
                                         w, data.width()));
  BitVector view = data.slice(0, w);
  // Materialize into byte vector sized to width. This can be shortcut by just
  // casting the view to a mutable bit vector, and grabbing its storage.
  MutableBitVector viewCopy(view);
  data >>= w;
  return std::any(viewCopy.takeStorage());
}

// Helper function to extract signed integer-like values from std::any.
// We here support cstdint int's and esi::Int.
static Int getIntLikeFromAny(const std::any &obj, unsigned widthHint) {
  const std::type_info &type = obj.type();

  if (type == typeid(int64_t))
    return Int(std::any_cast<int64_t>(obj), widthHint);
  if (type == typeid(int32_t))
    return Int(static_cast<int64_t>(std::any_cast<int32_t>(obj)), widthHint);
  if (type == typeid(int16_t))
    return Int(static_cast<int64_t>(std::any_cast<int16_t>(obj)), widthHint);
  if (type == typeid(int8_t))
    return Int(static_cast<int64_t>(std::any_cast<int8_t>(obj)), widthHint);
  if (type == typeid(esi::Int))
    return std::any_cast<Int>(obj);

  throw std::bad_any_cast();
}

// Helper function to extract unsigned integer-like values from std::any.
// We here support cstdint uint's and esi::UInt.
static UInt getUIntLikeFromAny(const std::any &obj, unsigned widthHint) {
  const std::type_info &type = obj.type();

  if (type == typeid(uint64_t))
    return UInt(std::any_cast<uint64_t>(obj), widthHint);
  if (type == typeid(uint32_t))
    return UInt(static_cast<uint64_t>(std::any_cast<uint32_t>(obj)), widthHint);
  if (type == typeid(uint16_t))
    return UInt(static_cast<uint64_t>(std::any_cast<uint16_t>(obj)), widthHint);
  if (type == typeid(uint8_t))
    return UInt(static_cast<uint64_t>(std::any_cast<uint8_t>(obj)), widthHint);
  if (type == typeid(esi::UInt))
    return std::any_cast<UInt>(obj);

  throw std::bad_any_cast();
}

void SIntType::ensureValid(const std::any &obj) const {
  try {
    Int value = getIntLikeFromAny(obj, getWidth());
  } catch (const std::exception &e) {
    throw std::runtime_error(std::format(
        "Unable to convert provided object to a {}-bit wide Int: {}",
        getWidth(), e.what()));
  }
}

MutableBitVector SIntType::serialize(const std::any &obj) const {
  Int ival = getIntLikeFromAny(obj, getWidth());
  if (static_cast<uint64_t>(ival.width()) != getWidth())
    throw std::runtime_error("Int width mismatch for SIntType serialize");
  // Move bits into MutableBitVector.
  return MutableBitVector(std::move(ival));
}

std::any SIntType::deserialize(BitVector &data) const {
  uint64_t w = getWidth();
  if (data.width() < w)
    throw std::runtime_error("Insufficient data for sint type");
  Int val(data.slice(0, w));
  data >>= w;
  return std::any(val);
}

void UIntType::ensureValid(const std::any &obj) const {
  try {
    UInt value = getUIntLikeFromAny(obj, getWidth());
  } catch (const std::exception &e) {
    throw std::runtime_error(std::format(
        "Unable to convert provided object to a {}-bit wide UInt: {}",
        getWidth(), e.what()));
  }
}

MutableBitVector UIntType::serialize(const std::any &obj) const {
  UInt uval = getUIntLikeFromAny(obj, getWidth());
  if (static_cast<uint64_t>(uval.width()) != getWidth())
    throw std::runtime_error("UInt width mismatch for UIntType serialize");
  return MutableBitVector(std::move(uval));
}

std::any UIntType::deserialize(BitVector &data) const {
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
  MutableBitVector out(0);
  auto handleField = [&](const std::pair<std::string, const Type *> &f) {
    const auto &name = f.first;
    const Type *ty = f.second;
    auto fieldData = ty->serialize(structData.at(name));
    out <<= fieldData.width();
    out |= fieldData;
  };
  if (isReverse()) {
    for (const auto &f : fields)
      handleField(f);
  } else {
    for (auto it = fields.rbegin(); it != fields.rend(); ++it)
      handleField(*it);
  }
  return out;
}

std::any StructType::deserialize(BitVector &data) const {
  std::map<std::string, std::any> result;
  auto consumeField = [&](const std::pair<std::string, const Type *> &f) {
    const auto &name = f.first;
    const Type *ty = f.second;
    result[name] = ty->deserialize(data);
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
  MutableBitVector out(0);
  auto handleElem = [&](const std::any &elem) {
    auto elemData = elementType->serialize(elem);
    out <<= elemData.width();
    out |= elemData;
  };
  if (isReverse()) {
    for (const auto &e : arrayData)
      handleElem(e);
  } else {
    for (auto it = arrayData.rbegin(); it != arrayData.rend(); ++it)
      handleElem(*it);
  }
  return out;
}

std::any ArrayType::deserialize(BitVector &data) const {
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
