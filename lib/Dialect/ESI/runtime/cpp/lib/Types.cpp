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

namespace esi {

//===----------------------------------------------------------------------===//
// ESI Type static 'create' methods
//===----------------------------------------------------------------------===//

const VoidType *VoidType::create(Context &ctxt,
                                 std::optional<Type::ID> typeID) {
  return ctxt.getOrCreateType<VoidType>(typeID.value_or("void"));
}

const Type *BitsType::create(Context &ctxt, uint64_t width,
                             std::optional<Type::ID> typeID) {
  if (width == 0)
    return VoidType::create(ctxt);
  return ctxt.getOrCreateType<BitsType>(
      typeID.value_or("bits" + std::to_string(width)), width);
}

const Type *SIntType::create(Context &ctxt, uint64_t width,
                             std::optional<Type::ID> typeID) {
  if (width == 0)
    return VoidType::create(ctxt);
  return ctxt.getOrCreateType<SIntType>(
      typeID.value_or(std::format("si{}", width)), width);
}

const Type *UIntType::create(Context &ctxt, uint64_t width,
                             std::optional<Type::ID> typeID) {
  if (width == 0)
    return VoidType::create(ctxt);
  return ctxt.getOrCreateType<UIntType>(
      typeID.value_or(std::format("ui{}", width)), width);
}

const AnyType *AnyType::create(Context &ctxt) {
  return ctxt.getOrCreateType<AnyType>("any");
}

const StructType *StructType::create(Context &ctxt, const FieldVector &fields,
                                     bool reverse,
                                     std::optional<Type::ID> typeID) {
  if (!typeID.has_value()) {
    // Infer type ID from field names and types.
    typeID = "struct{";
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i > 0)
        *typeID += ",";
      *typeID +=
          std::format("{}:{}", fields[i].first, fields[i].second->getID());
    }
    *typeID += "}";
  }
  return ctxt.getOrCreateType<StructType>(*typeID, fields, reverse);
}

const ArrayType *ArrayType::create(Context &ctxt, const Type *elementType,
                                   uint64_t size, bool reverse,
                                   std::optional<Type::ID> typeID) {
  return ctxt.getOrCreateType<ArrayType>(
      typeID.value_or(std::format("array_{}_{}", size, elementType->getID())),
      elementType, size, reverse);
}

const ChannelType *ChannelType::create(Context &ctxt, const Type *inner,
                                       std::optional<Type::ID> typeID) {
  return ctxt.getOrCreateType<ChannelType>(
      typeID.value_or(std::format("channel_{}", inner->getID())), inner);
}

const BundleType *BundleType::create(Context &ctxt,
                                     const ChannelVector &channels,
                                     std::optional<Type::ID> typeID) {
  Type::ID id;
  if (typeID) {
    id = *typeID;
  } else {
    // Infer type ID from channel names, directions, and types.
    id = "bundle{";
    for (size_t i = 0; i < channels.size(); ++i) {
      if (i > 0)
        id += ",";
      const auto &[name, dir, type] = channels[i];
      id += (dir == BundleType::To ? "to:" : "from:") + name + ":" +
            type->getID();
    }
    id += "}";
  }
  return ctxt.getOrCreateType<BundleType>(id, channels);
}

//===----------------------------------------------------------------------===//
// Type dump / toString
//===----------------------------------------------------------------------===//

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

// Helper function to deserialize a single type from the type table.
// NOLINTNEXTLINE(misc-no-recursion)
static const Type *deserializeTypeFromTable(esi::Context &ctx,
                                            const nlohmann::json &typeTable,
                                            const Type::ID &typeID) {
  // Check if already registered in context (handles caching/cycles).
  std::optional<const esi::Type *> existing = ctx.getType(typeID);
  if (existing)
    return existing.value();

  if (!typeTable.contains(typeID))
    throw std::runtime_error(
        std::format("Type ID '{}' not found in type table", typeID));

  const nlohmann::json &j = typeTable.at(typeID);

  if (!j.is_object())
    throw std::runtime_error("deserializeTypeFromTable: expected JSON object");

  std::string mnemonic = j.at("mnemonic").get<std::string>();
  // Extract optional type ID from JSON (nullopt if empty or missing).
  std::optional<Type::ID> id;
  if (j.contains("id") && !j.at("id").get<std::string>().empty())
    id = j.at("id").get<std::string>();

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
    if (signedness == "signed")
      return SIntType::create(ctx, width, id);
    return UIntType::create(ctx, width, id);
  }
  if (mnemonic == "uint") {
    // Legacy format support
    uint64_t width = getBitwidth();
    return UIntType::create(ctx, width, id);
  }
  if (mnemonic == "sint") {
    // Legacy format support
    uint64_t width = getBitwidth();
    return SIntType::create(ctx, width, id);
  }
  if (mnemonic == "bits") {
    uint64_t width = getBitwidth();
    return BitsType::create(ctx, width, id);
  }
  if (mnemonic == "void") {
    return VoidType::create(ctx);
  }
  if (mnemonic == "any") {
    return AnyType::create(ctx);
  }
  if (mnemonic == "struct") {
    esi::StructType::FieldVector fields;
    for (const auto &fieldObj : j.at("fields")) {
      std::string fieldName = fieldObj.at("name").get<std::string>();
      Type::ID fieldTypeID = fieldObj.at("typeID").get<std::string>();
      const esi::Type *fieldType =
          deserializeTypeFromTable(ctx, typeTable, fieldTypeID);
      fields.emplace_back(fieldName, fieldType);
    }
    // Use the serialized ID if present, otherwise let create() infer one.
    return StructType::create(ctx, fields, /*reverse=*/true, id);
  }
  if (mnemonic == "array") {
    Type::ID elementTypeID = j.at("element").get<std::string>();
    const esi::Type *elementType =
        deserializeTypeFromTable(ctx, typeTable, elementTypeID);
    uint64_t size = j.at("size").get<uint64_t>();
    return ArrayType::create(ctx, elementType, size, /*reverse=*/true, id);
  }
  if (mnemonic == "channel") {
    Type::ID innerTypeID = j.at("inner").get<std::string>();
    const esi::Type *innerType =
        deserializeTypeFromTable(ctx, typeTable, innerTypeID);
    return ChannelType::create(ctx, innerType, id);
  }
  if (mnemonic == "bundle") {
    esi::BundleType::ChannelVector channels;
    for (const auto &channelObj : j.at("channels")) {
      std::string channelName = channelObj.at("name").get<std::string>();
      std::string dirStr = channelObj.at("direction").get<std::string>();
      esi::BundleType::Direction direction =
          (dirStr == "to") ? esi::BundleType::Direction::To
                           : esi::BundleType::Direction::From;
      Type::ID channelTypeID = channelObj.at("typeID").get<std::string>();
      const esi::Type *channelType =
          deserializeTypeFromTable(ctx, typeTable, channelTypeID);
      channels.emplace_back(channelName, direction, channelType);
    }
    // Use the serialized ID if present, otherwise let create() infer one.
    return BundleType::create(ctx, channels, id);
  }
  throw std::runtime_error(
      std::format("deserializeTypeFromTable: unknown mnemonic '{}'", mnemonic));
}

// Deserialize the provided string to a type instance. The type (and any
// nested types) are registered to the provided context. It is expected that
// the type was serialized via the 'serialize' method.
const Type *Type::deserializeType(esi::Context &ctx, const std::string &data) {
  nlohmann::json j = nlohmann::json::parse(data);

  if (!j.is_object())
    throw std::runtime_error("deserializeType: expected JSON object");

  if (!j.contains("typeTable"))
    throw std::runtime_error("deserializeType: missing typeTable field");

  if (!j.contains("typeID"))
    throw std::runtime_error("deserializeType: missing typeID field");

  const nlohmann::json &typeTable = j.at("typeTable");
  Type::ID rootTypeID = j.at("typeID").get<std::string>();
  return deserializeTypeFromTable(ctx, typeTable, rootTypeID);
}

// Serializes the provided type into the provided type table object, and returns
// the type ID of the serialized type.
// NOLINTNEXTLINE(misc-no-recursion)
static esi::Type::ID serializeTypeToTable(const esi::Type *type,
                                          nlohmann::json &table) {
  // Check whether we've already visited this type. If so, return the typeID
  // directly.
  esi::Type::ID typeID = type->getID();
  if (table.contains(typeID))
    return typeID;

  nlohmann::json j;
  j["id"] = typeID;

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
      fieldObj["typeID"] = serializeTypeToTable(fieldType, table);
      fieldsArr.push_back(fieldObj);
    }
    j["fields"] = fieldsArr;
  } else if (auto *arrayType = dynamic_cast<const esi::ArrayType *>(type)) {
    j["mnemonic"] = "array";
    j["element"] = serializeTypeToTable(arrayType->getElementType(), table);
    j["size"] = arrayType->getSize();
  } else if (auto *channelType = dynamic_cast<const esi::ChannelType *>(type)) {
    j["mnemonic"] = "channel";
    j["inner"] = serializeTypeToTable(channelType->getInner(), table);
  } else if (auto *bundleType = dynamic_cast<const esi::BundleType *>(type)) {
    j["mnemonic"] = "bundle";
    nlohmann::json channelsArr = nlohmann::json::array();
    for (const auto &[name, direction, channelType] :
         bundleType->getChannels()) {
      nlohmann::json channelObj;
      channelObj["name"] = name;
      channelObj["direction"] =
          (direction == esi::BundleType::Direction::To) ? "to" : "from";
      channelObj["typeID"] = serializeTypeToTable(channelType, table);
      channelsArr.push_back(channelObj);
    }
    j["channels"] = channelsArr;
  } else {
    throw std::runtime_error(
        std::format("typeToJsonObj: unhandled type '{}'", type->getID()));
  }

  // Inject into the type table.
  table[type->getID()] = j;

  return type->getID();
}

// NOLINTNEXTLINE(misc-no-recursion)
std::string Type::serializeType(const esi::Type *type) {
  if (!type)
    throw std::runtime_error("Type::serializeType: null type provided");

  nlohmann::json j;
  // Initialize the type table, which is a mapping from type IDs to serialized
  // type objects.
  auto &table = j["typeTable"];
  j["typeID"] = serializeTypeToTable(type, table);

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
