//===- ESICapnp.cpp - ESI Cap'nProto utilities ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ESICapnp.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/IndentingOStream.h"

#include "capnp/schema-parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

namespace circt {
namespace esi {
namespace capnp {

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

/// Emit an ID in capnp format.
llvm::raw_ostream &emitId(llvm::raw_ostream &os, int64_t id) {
  return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
}

/// Write a valid Capnp name for 'type'.
static void emitName(Type type, uint64_t id, llvm::raw_ostream &os) {
  llvm::TypeSwitch<Type>(type)
      .Case([&os](IntegerType intTy) {
        std::string intName;
        llvm::raw_string_ostream(intName) << intTy;
        // Capnp struct names must start with an uppercase character.
        intName[0] = toupper(intName[0]);
        os << intName;
      })
      .Case([&os](hw::ArrayType arrTy) {
        os << "ArrayOf" << arrTy.getSize() << 'x';
        emitName(arrTy.getElementType(), 0, os);
      })
      .Case([&os](NoneType) { os << "None"; })
      .Case([&os, id](hw::StructType t) { os << "Struct" << id; })
      .Default([](Type) {
        assert(false && "Type not supported. Please check support first with "
                        "isSupported()");
      });
}

/// Returns true if the type is currently supported.
static bool isSupported(Type type, bool outer = false) {
  return llvm::TypeSwitch<::mlir::Type, bool>(type)
      .Case([](IntegerType t) { return t.getWidth() <= 64; })
      .Case([](hw::ArrayType t) { return isSupported(t.getElementType()); })
      .Case([outer](hw::StructType t) {
        // We don't yet support structs containing structs.
        if (!outer)
          return false;
        // A struct is supported if all of its elements are.
        for (auto field : t.getElements()) {
          if (!isSupported(field.type))
            return false;
        }
        return true;
      })
      .Default([](Type) { return false; });
}

bool ESICapnpType::isSupported() const {
  return esi::capnp::isSupported(type, true);
}

circt::esi::capnp::ESICapnpType::ESICapnpType(Type _type) : type(_type) {
  type = innerType(type);

  TypeSwitch<Type>(type)
      .Case([this](IntegerType t) {
        fieldTypes.push_back(
            FieldInfo{StringAttr::get(t.getContext(), "i"), t});
      })
      .Case([this](hw::ArrayType t) {
        fieldTypes.push_back(
            FieldInfo{StringAttr::get(t.getContext(), "l"), t});
      })
      .Case([this](hw::StructType t) {
        fieldTypes.append(t.getElements().begin(), t.getElements().end());
      })
      .Default([](Type) {});
}

bool ESICapnpType::operator==(const ESICapnpType &that) const {
  return type == that.type;
}

/// For now, the name is just the type serialized. This works only because we
/// only support ints.
StringRef ESICapnpType::capnpName() const {
  if (cachedName == "") {
    llvm::raw_string_ostream os(cachedName);
    emitName(type, capnpTypeID(), os);
    cachedName = os.str();
  }
  return cachedName;
}

// We compute a deterministic hash based on the type. Since llvm::hash_value
// changes from execution to execution, we don't use it.
uint64_t ESICapnpType::capnpTypeID() const {
  if (cachedID)
    return *cachedID;

  // Get the MLIR asm type, padded to a multiple of 64 bytes.
  std::string typeName;
  llvm::raw_string_ostream osName(typeName);
  osName << type;
  size_t overhang = osName.tell() % 64;
  if (overhang != 0)
    osName.indent(64 - overhang);
  osName.flush();
  const char *typeNameC = typeName.c_str();

  uint64_t hash = esiCosimSchemaVersion;
  for (size_t i = 0, e = typeName.length() / 64; i < e; ++i)
    hash =
        llvm::hashing::detail::hash_33to64_bytes(&typeNameC[i * 64], 64, hash);

  // Capnp IDs always have a '1' high bit.
  cachedID = hash | 0x8000000000000000;
  return *cachedID;
}

void ESICapnpType::writeMetadata(llvm::raw_ostream &os) const {
  os << capnpName() << " ";
  emitId(os, capnpTypeID());
}

} // namespace capnp
} // namespace esi
} // namespace circt
