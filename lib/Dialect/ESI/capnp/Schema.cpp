//===- Schema.cpp - ESI Cap'nProto schema utilities -------------*- C++ -*-===//
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
#include "circt/Dialect/ESI/ESITypes.h"

#include "capnp/schema-parser.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

using namespace mlir;
using namespace circt::esi::capnp::detail;

namespace {
/// Intentation utils.
class IndentingOStream {
public:
  IndentingOStream(llvm::raw_ostream &os) : os(os) {}

  template <typename T>
  IndentingOStream &operator<<(T t) {
    os << t;
    return *this;
  }

  IndentingOStream &indent() {
    os.indent(currentIndent);
    return *this;
  }
  void addIndent() { currentIndent += 2; }
  void reduceIndent() { currentIndent -= 2; }

private:
  llvm::raw_ostream &os;
  size_t currentIndent = 0;
};
} // namespace

namespace circt {
namespace esi {
namespace capnp {
namespace detail {
struct TypeSchemaStorage {
public:
  TypeSchemaStorage(Type type) : type(type) {}

  /// Get the Cap'nProto schema ID for a type.
  uint64_t capnpTypeID() const;

  bool isSupported() const;
  size_t size() const;
  StringRef name() const;
  mlir::LogicalResult write(llvm::raw_ostream &os);
  mlir::LogicalResult writeMetadata(llvm::raw_ostream &os);

  bool operator==(const TypeSchemaStorage &) const;

private:
  ::capnp::ParsedSchema getSchema() const;
  ::capnp::StructSchema getStructSchema() const;

  Type type;
  mutable std::string cachedName;

  ::capnp::ParsedSchema schema;
};
} // namespace detail
} // namespace capnp
} // namespace esi
} // namespace circt

// ::capnp::ParsedSchema getSchema();

::capnp::StructSchema TypeSchemaStorage::getStructSchema() const {
  uint64_t id = capnpTypeID();
  for (auto schemaNode : getSchema().getAllNested()) {
    if (schemaNode.getProto().getId() == id)
      return schemaNode.asStruct();
  }
  assert(false && "A node with a matching ID should always be found.");
}

// We compute a deterministic hash based on the type. Since llvm::hash_value
// changes from execution to execution, we don't use it. This assumes a closed
// type system, which is reasonable since we only support some types in the
// Capnp schema generation anyway.
uint64_t TypeSchemaStorage::capnpTypeID() const {
  // We can hash up to 64 bytes with a single function call.
  char buffer[64];
  memset(buffer, 0, sizeof(buffer));

  // The first byte is for the outer type.
  buffer[0] = 1; // Constant for the ChannelPort type.

  TypeSwitch<Type>(type)
      .Case([&buffer](IntegerType t) {
        // The second byte is for the inner type.
        buffer[1] = 1;
        // The rest can be defined arbitrarily.
        buffer[2] = (char)t.getSignedness();
        *(int64_t *)&buffer[4] = t.getWidth();
      })
      .Default([](Type) { assert(false && "Type not yet supported"); });

  uint64_t hash =
      llvm::hashing::detail::hash_short(buffer, 12, esiCosimSchemaVersion);
  // Capnp IDs always have a '1' high bit.
  return hash | 0x8000000000000000;
}

bool TypeSchemaStorage::isSupported() const {
  return llvm::TypeSwitch<::mlir::Type, bool>(type)
      .Case<IntegerType>([](IntegerType t) { return t.getWidth() <= 64; })
      .Default([](Type) { return false; });
}

// Compute the expected size of the capnp message in bits.
size_t TypeSchemaStorage::size() const {
  auto schema = getStructSchema();
  auto structProto = schema.getProto().getStruct();
  return 64 * (structProto.getDataWordCount() + structProto.getPointerCount());
}

StringRef TypeSchemaStorage::name() const {
  if (cachedName == "") {
    llvm::raw_string_ostream os(cachedName);
    os << "TY" << type;
    cachedName = os.str();
  }
  return cachedName;
}

/// Emit an ID in capnp format.
static llvm::raw_ostream &emitId(llvm::raw_ostream &os, int64_t id) {
  return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
}

mlir::LogicalResult TypeSchemaStorage::write(llvm::raw_ostream &rawOS) {
  IndentingOStream os(rawOS);

  // Since capnp requires messages to be structs, emit a wrapper struct.
  os.indent() << "struct ";
  writeMetadata(rawOS);
  os << " {\n";
  os.addIndent();

  auto intTy = type.dyn_cast<IntegerType>();
  assert(intTy &&
         "Type not supported. Please check support first with isSupported()");

  // Specify the actual type, followed by the capnp field.
  os.indent() << "# Actual type is " << type << ".\n";
  os.indent() << "i @0 :";

  auto w = intTy.getWidth();
  if (w == 1) {
    os.indent() << "Bool";
  } else {
    if (intTy.isSigned())
      os << "Int";
    else
      os << "UInt";

    // Round up.
    if (w <= 8)
      os << "8";
    else if (w <= 16)
      os << "16";
    else if (w <= 32)
      os << "32";
    else if (w <= 64)
      os << "64";
    else
      assert(false && "Type not supported. Please check support first with "
                      "isSupported()");
  }
  os << ";\n";

  os.reduceIndent();
  os.indent() << "}\n\n";
  return success();
}

mlir::LogicalResult TypeSchemaStorage::writeMetadata(llvm::raw_ostream &os) {
  os << name() << " ";
  emitId(os, capnpTypeID());
  return success();
}

bool TypeSchemaStorage::operator==(const TypeSchemaStorage &that) const {
  return type == that.type;
}

//===----------------------------------------------------------------------===//
// TypeSchema wrapper.
//===----------------------------------------------------------------------===//

circt::esi::capnp::TypeSchema::TypeSchema(Type type) {
  circt::esi::ChannelPort chan = type.dyn_cast<circt::esi::ChannelPort>();
  if (chan)
    type = chan.getInner();
  s = std::make_shared<detail::TypeSchemaStorage>(type);
}
uint64_t circt::esi::capnp::TypeSchema::capnpTypeID() const {
  return s->capnpTypeID();
}

bool circt::esi::capnp::TypeSchema::isSupported() const {
  return s->isSupported();
}

// Compute the expected size of the capnp message in bits.
size_t circt::esi::capnp::TypeSchema::size() const { return s->size(); }

StringRef circt::esi::capnp::TypeSchema::name() const { return s->name(); }

mlir::LogicalResult
circt::esi::capnp::TypeSchema::write(llvm::raw_ostream &os) {
  return s->write(os);
}

mlir::LogicalResult
circt::esi::capnp::TypeSchema::writeMetadata(llvm::raw_ostream &os) {
  return s->writeMetadata(os);
}

bool circt::esi::capnp::TypeSchema::operator==(const TypeSchema &that) const {
  return *s == *that.s;
}
