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
#include "esi/Context.h"
#include "esi/Values.h"
#include <algorithm>
#include <cstring>
#include <format>
#include <span>
#include <sstream>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcovered-switch-default"
#endif
#include <nlohmann/json.hpp>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

//===----------------------------------------------------------------------===//
// ESI Type Construction Helpers
//===----------------------------------------------------------------------===//

/// Generic helper to check for existing types in the context and register new
/// ones.
template <typename T>
static const T *getOrCreateType(esi::Context &ctxt, const std::string &typeName,
                                std::function<T *()> createFunc) {
  std::optional<const esi::Type *> existing = ctxt.getType(typeName);
  if (existing) {
    const auto *existingTyped = dynamic_cast<const T *>(existing.value());
    if (!existingTyped) {
      throw std::runtime_error(std::format(
          "Type ID '{}' already registered with a different type", typeName));
    }
    return existingTyped;
  }

  auto *newType = createFunc();
  ctxt.registerType(newType);
  return newType;
}

static const esi::VoidType *createVoidType(esi::Context &ctxt) {
  std::string typeName = "void";
  return getOrCreateType<esi::VoidType>(
      ctxt, typeName, [&typeName]() { return new esi::VoidType(typeName); });
}

static const esi::Type *createBitsType(esi::Context &ctxt, size_t bitWidth) {
  if (bitWidth == 0)
    return createVoidType(ctxt);

  std::string typeName = "bits" + std::to_string(bitWidth);
  return getOrCreateType<esi::BitsType>(
      ctxt, typeName, [&typeName, bitWidth]() {
        return new esi::BitsType(typeName, bitWidth);
      });
}

static const esi::Type *createIntType(esi::Context &ctxt, size_t bitWidth,
                                      bool isSigned) {
  if (bitWidth == 0)
    return createVoidType(ctxt);

  std::string typeName = std::format("{}i{}", isSigned ? "s" : "u", bitWidth);

  if (isSigned) {
    return getOrCreateType<esi::SIntType>(
        ctxt, typeName, [&typeName, bitWidth]() {
          return new esi::SIntType(typeName, bitWidth);
        });
  }

  return getOrCreateType<esi::UIntType>(
      ctxt, typeName, [&typeName, bitWidth]() {
        return new esi::UIntType(typeName, bitWidth);
      });
}

static const esi::Type *createUIntType(esi::Context &ctxt, size_t bitWidth) {
  return createIntType(ctxt, bitWidth, false);
}

static const esi::Type *createSIntType(esi::Context &ctxt, size_t bitWidth) {
  return createIntType(ctxt, bitWidth, true);
}

static const esi::ArrayType *createArrayType(esi::Context &ctxt,
                                             const esi::Type *elementType,
                                             size_t size, bool reverse = true) {
  std::string typeName =
      "array_" + std::to_string(size) + "_" + elementType->getID();

  return getOrCreateType<esi::ArrayType>(
      ctxt, typeName, [&typeName, elementType, size, reverse]() {
        return new esi::ArrayType(typeName, elementType, size, reverse);
      });
}

static const esi::StructType *
createStructType(esi::Context &ctxt, const std::string &name,
                 const esi::StructType::FieldVector &fields,
                 bool reverse = true) {
  return getOrCreateType<esi::StructType>(
      ctxt, name, [&name, &fields, reverse]() {
        return new esi::StructType(name, fields, reverse);
      });
}

static const esi::ChannelType *createChannelType(esi::Context &ctxt,
                                                 const esi::Type *inner) {
  std::string channelTypeName = "channel_" + inner->getID();
  return getOrCreateType<esi::ChannelType>(
      ctxt, channelTypeName, [&channelTypeName, inner]() {
        return new esi::ChannelType(channelTypeName, inner);
      });
}

static const esi::BundleType *
createFunctionBundleType(esi::Context &ctxt, const std::string &name,
                         const esi::Type *argType,
                         const esi::Type *resultType) {
  esi::BundleType::ChannelVector channels;
  channels.emplace_back("arg", esi::BundleType::Direction::To, argType);
  channels.emplace_back("result", esi::BundleType::Direction::From, resultType);
  auto *bundleType = new esi::BundleType(name, channels);
  ctxt.registerType(bundleType);
  return bundleType;
}

namespace esi {

// NOLINTNEXTLINE(misc-no-recursion)
static void dumpType(std::ostream &os, const esi::Type *type, int level = 0,
                     bool oneLine = false) {
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
    os << "struct {";
    const auto &fields = structType->getFields();
    for (size_t i = 0; i < fields.size(); ++i) {
      if (!oneLine)
        os << std::endl << std::string(level + 2, ' ');
      else if (i > 0)
        os << " ";
      const auto &[name, fieldType] = fields[i];
      os << name << ": ";
      dumpType(os, fieldType, level + 1, oneLine);
      if (i < fields.size() - 1)
        os << ",";
    }
    if (!oneLine)
      os << std::endl << std::string(level, ' ');
    os << "}";
  } else if (auto *arrayType = dynamic_cast<const esi::ArrayType *>(type)) {
    dumpType(os, arrayType->getElementType(), level + 1, oneLine);
    os << "[" << arrayType->getSize() << "]";
  } else if (auto *channelType = dynamic_cast<const esi::ChannelType *>(type)) {
    os << "chan<";
    dumpType(os, channelType->getInner(), level + 1, oneLine);
    os << ">";
  } else if (auto *bundleType = dynamic_cast<const esi::BundleType *>(type)) {
    os << "bundle {";
    const auto &channels = bundleType->getChannels();
    for (size_t i = 0; i < channels.size(); ++i) {
      if (!oneLine)
        os << std::endl << std::string(level + 2, ' ');
      else if (i > 0)
        os << " ";
      const auto &[name, direction, fieldType] = channels[i];
      os << std::format(
          "{} [{}]: ", direction == BundleType::To ? "to" : "from", name);
      dumpType(os, fieldType, level + 1, oneLine);
      if (i < channels.size() - 1)
        os << ",";
    }
    if (!oneLine)
      os << std::endl << std::string(level, ' ');
    else
      os << " ";
    os << "}";
  } else if (auto *windowType = dynamic_cast<const esi::WindowType *>(type)) {
    os << "window[";
    dumpType(os, windowType->getIntoType(), level + 1, oneLine);
    os << "]";
  } else if (auto *listType = dynamic_cast<const esi::ListType *>(type)) {
    os << "list<";
    dumpType(os, listType->getElementType(), level + 1, oneLine);
    os << ">";
  } else {
    // For unknown types, just print the type ID
    os << type->getID();
  }
}

void Type::dump(std::ostream &os, bool oneLine) const {
  dumpType(os, this, oneLine ? 0 : 0, oneLine);
}

// Deserialize the provided string to a type instance. The type (and any
// nested types) are registered to the provided context. It is expected that
// the type was serialized via the 'serialize' method.
// NOLINTNEXTLINE(misc-no-recursion)
const Type *Type::deserialize(esi::Context &ctx, const std::string &data) {
  nlohmann::json j = nlohmann::json::parse(data);

  if (!j.is_object())
    throw std::runtime_error("jsonObjToType: expected JSON object");

  std::string mnemonic = j.at("mnemonic").get<std::string>();
  std::string id = j.value("id", "");

  // Helper to get bitwidth from either hwBitwidth or width field
  auto getBitwidth = [&j]() -> uint64_t {
    if (j.contains("hwBitwidth"))
      return j.at("hwBitwidth").get<uint64_t>();
    return j.at("width").get<uint64_t>();
  };

  // Handle unified "int" mnemonic with signedness field
  if (mnemonic == "int") {
    uint64_t width = getBitwidth();
    std::string signedness = j.value("signedness", "unsigned");
    return createIntType(ctx, width, signedness == "signed");
  }
  if (mnemonic == "uint") {
    // Legacy format support
    uint64_t width = getBitwidth();
    return createUIntType(ctx, width);
  }
  if (mnemonic == "sint") {
    // Legacy format support
    uint64_t width = getBitwidth();
    return createSIntType(ctx, width);
  }
  if (mnemonic == "bits") {
    uint64_t width = getBitwidth();
    return createBitsType(ctx, width);
  }
  if (mnemonic == "void") {
    return createVoidType(ctx);
  }
  if (mnemonic == "any") {
    // AnyType doesn't have a create helper, create directly
    std::string typeName = id.empty() ? "any" : id;
    std::optional<const esi::Type *> existing = ctx.getType(typeName);
    if (existing)
      return existing.value();
    auto *anyType = new esi::AnyType(typeName);
    ctx.registerType(anyType);
    return anyType;
  }
  if (mnemonic == "struct") {
    esi::StructType::FieldVector fields;
    for (const auto &fieldObj : j.at("fields")) {
      std::string fieldName = fieldObj.at("name").get<std::string>();
      const esi::Type *fieldType = deserialize(ctx, fieldObj.at("type"));
      fields.emplace_back(fieldName, fieldType);
    }
    std::string typeName = id.empty() ? "struct_auto" : id;
    return createStructType(ctx, typeName, fields);
  }
  if (mnemonic == "array") {
    const esi::Type *elementType = deserialize(ctx, j.at("element"));
    uint64_t size = j.at("size").get<uint64_t>();
    return createArrayType(ctx, elementType, size);
  }
  if (mnemonic == "channel") {
    const esi::Type *innerType = deserialize(ctx, j.at("inner"));
    return createChannelType(ctx, innerType);
  }
  if (mnemonic == "bundle") {
    esi::BundleType::ChannelVector channels;
    for (const auto &channelObj : j.at("channels")) {
      std::string channelName = channelObj.at("name").get<std::string>();
      std::string dirStr = channelObj.at("direction").get<std::string>();
      esi::BundleType::Direction direction =
          (dirStr == "to") ? esi::BundleType::Direction::To
                           : esi::BundleType::Direction::From;
      const esi::Type *channelType = deserialize(ctx, channelObj.at("type"));
      channels.emplace_back(channelName, direction, channelType);
    }
    std::string typeName = id.empty() ? "bundle_auto" : id;
    // Check for existing type first
    std::optional<const esi::Type *> existing = ctx.getType(typeName);
    if (existing)
      return existing.value();
    auto *bundleType = new esi::BundleType(typeName, channels);
    ctx.registerType(bundleType);
    return bundleType;
  }
  throw std::runtime_error(
      std::format("jsonObjToType: unknown mnemonic '{}'", mnemonic));
}

// NOLINTNEXTLINE(misc-no-recursion)
std::string Type::serialize(const esi::Type *type) {
  if (!type)
    throw std::runtime_error("Type::serialize: null type provided");

  nlohmann::json j;
  j["id"] = type->getID();

  if (auto *uintType = dynamic_cast<const esi::UIntType *>(type)) {
    j["mnemonic"] = "int";
    j["hwBitwidth"] = uintType->getWidth();
    j["signedness"] = "unsigned";
  } else if (auto *sintType = dynamic_cast<const esi::SIntType *>(type)) {
    j["mnemonic"] = "int";
    j["hwBitwidth"] = sintType->getWidth();
    j["signedness"] = "signed";
  } else if (auto *bitsType = dynamic_cast<const esi::BitsType *>(type)) {
    j["mnemonic"] = "bits";
    j["hwBitwidth"] = bitsType->getWidth();
  } else if (dynamic_cast<const esi::VoidType *>(type)) {
    j["mnemonic"] = "void";
  } else if (dynamic_cast<const esi::AnyType *>(type)) {
    j["mnemonic"] = "any";
  } else if (auto *structType = dynamic_cast<const esi::StructType *>(type)) {
    j["mnemonic"] = "struct";
    nlohmann::json fieldsArr = nlohmann::json::array();
    for (const auto &[name, fieldType] : structType->getFields()) {
      nlohmann::json fieldObj;
      fieldObj["name"] = name;
      fieldObj["type"] = serialize(fieldType);
      fieldsArr.push_back(fieldObj);
    }
    j["fields"] = fieldsArr;
  } else if (auto *arrayType = dynamic_cast<const esi::ArrayType *>(type)) {
    j["mnemonic"] = "array";
    j["element"] = serialize(arrayType->getElementType());
    j["size"] = arrayType->getSize();
  } else if (auto *channelType = dynamic_cast<const esi::ChannelType *>(type)) {
    j["mnemonic"] = "channel";
    j["inner"] = serialize(channelType->getInner());
  } else if (auto *bundleType = dynamic_cast<const esi::BundleType *>(type)) {
    j["mnemonic"] = "bundle";
    nlohmann::json channelsArr = nlohmann::json::array();
    for (const auto &[name, direction, channelType] :
         bundleType->getChannels()) {
      nlohmann::json channelObj;
      channelObj["name"] = name;
      channelObj["direction"] =
          (direction == esi::BundleType::Direction::To) ? "to" : "from";
      channelObj["type"] = serialize(channelType);
      channelsArr.push_back(channelObj);
    }
    j["channels"] = channelsArr;
  } else {
    throw std::runtime_error(
        std::format("typeToJsonObj: unhandled type '{}'", type->getID()));
  }

  return j.dump();
}

// Recurse through an ESI type and print an elaborated version of it.
std::string Type::toString(bool oneLine) const {
  std::stringstream ss;
  dump(ss, oneLine);
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
