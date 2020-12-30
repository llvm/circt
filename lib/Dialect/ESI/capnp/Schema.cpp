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

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

using namespace mlir;
using namespace circt::esi;
using namespace circt::esi::capnp;

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

  Type type;
  std::string name;
};
} // namespace detail
} // namespace capnp
} // namespace esi
} // namespace circt

TypeSchema::TypeSchema(Type type) {
  ChannelPort chan = type.dyn_cast<ChannelPort>();
  if (chan)
    type = chan.getInner();
  s = std::make_shared<detail::TypeSchemaStorage>(type);
}

// We compute a deterministic hash based on the type. Since llvm::hash_value
// changes from execution to execution, we don't use it. This assumes a closed
// type system, which is reasonable since we only support some types in the
// Capnp schema generation anyway.
uint64_t TypeSchema::capnpTypeID() const {
  // We can hash up to 64 bytes with a single function call.
  char buffer[64];
  memset(buffer, 0, sizeof(buffer));

  // The first byte is for the outer type.
  buffer[0] = 1; // Constant for the ChannelPort type.

  TypeSwitch<Type>(s->type)
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

bool TypeSchema::isSupported() const {
  return llvm::TypeSwitch<::mlir::Type, bool>(s->type)
      .Case<IntegerType>([](IntegerType t) { return t.getWidth() <= 64; })
      .Default([](Type) { return false; });
}

// Compute the expected size of the capnp message field in bits.
static size_t getCapnpMsgFieldSize(Type type) {
  return llvm::TypeSwitch<::mlir::Type, size_t>(type)
      .Case<IntegerType>([](IntegerType t) {
        auto w = t.getWidth();
        if (w == 1)
          return 8;
        else if (w <= 8)
          return 8;
        else if (w <= 16)
          return 16;
        else if (w <= 32)
          return 32;
        else if (w <= 64)
          return 64;
        return -1;
      })
      .Default([](Type) {
        assert(false && "Type not supported. Please check support first with "
                        "isSupported()");
        return 0;
       });
}

// Compute the expected size of the capnp message in bits. Return -1 on
// non-representable type. TODO: replace this with a call into the Capnp C++
// library to parse a schema and have it compute sizes and offsets.
size_t TypeSchema::size() const {
  size_t headerSize = 128;
  size_t fieldSize = getCapnpMsgFieldSize(s->type);
  // Capnp sizes are always multiples of 8 bytes, so round up.
  fieldSize = (fieldSize & ~0x3f) + (fieldSize & 0x3f ? 0x40 : 0);
  return headerSize + fieldSize;
}

StringRef TypeSchema::name() const {
  if (s->name == "") {
    llvm::raw_string_ostream os(s->name);
    os << "TY" << s->type;
  }
  return s->name;
}

/// Emit an ID in capnp format.
llvm::raw_ostream &emitId(llvm::raw_ostream &os, int64_t id) {
  return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
}

mlir::LogicalResult TypeSchema::write(llvm::raw_ostream &raw_os) {
  IndentingOStream os(raw_os);

  // Since capnp requires messages to be structs, emit a wrapper struct.
  os.indent() << "struct ";
  writeMetadata(raw_os);
  os << " {\n";
  os.addIndent();

  auto intTy = s->type.dyn_cast<IntegerType>();
  assert(intTy &&
         "Type not supported. Please check support first with isSupported()");

  // Specify the actual type, followed by the capnp field.
  os.indent() << "# Actual type is " << s->type << ".\n";
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

mlir::LogicalResult TypeSchema::writeMetadata(llvm::raw_ostream &os) {
  os << name() << " ";
  emitId(os, capnpTypeID());
  return success();
}

bool TypeSchema::operator==(const TypeSchema &that) const {
  return s->type == that.s->type;
}
