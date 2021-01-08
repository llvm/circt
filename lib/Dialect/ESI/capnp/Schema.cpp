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
#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/RTL/RTLTypes.h"
#include "circt/Dialect/SV/SVOps.h"

#include "capnp/schema-parser.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

#include <string>

using namespace mlir;
using namespace circt::esi::capnp::detail;
using namespace circt;

//===----------------------------------------------------------------------===//
// Utilities.
//===----------------------------------------------------------------------===//

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

/// Emit an ID in capnp format.
static llvm::raw_ostream &emitId(llvm::raw_ostream &os, int64_t id) {
  return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
}

//===----------------------------------------------------------------------===//
// TypeSchema class implementation.
//===----------------------------------------------------------------------===//

namespace circt {
namespace esi {
namespace capnp {
namespace detail {
/// Actual implementation of `TypeSchema` to keep all the details out of the
/// header.
struct TypeSchemaImpl {
public:
  TypeSchemaImpl(Type type) : type(type) {}

  mlir::Type getType() const { return type; }

  uint64_t capnpTypeID() const;

  bool isSupported() const;
  size_t size() const;
  StringRef name() const;
  LogicalResult write(llvm::raw_ostream &os) const;
  LogicalResult writeMetadata(llvm::raw_ostream &os) const;

  bool operator==(const TypeSchemaImpl &) const;

  /// Build an RTL/SV dialect capnp encoder for this type.
  Value buildEncoder(OpBuilder &, Value);
  /// Build an RTL/SV dialect capnp decoder for this type.
  Value buildDecoder(OpBuilder &, Value);

private:
  ::capnp::ParsedSchema getSchema() const;
  ::capnp::StructSchema getTypeSchema() const;

  Type type;
  ::capnp::SchemaParser parser;
  mutable llvm::Optional<uint64_t> cachedID;
  mutable std::string cachedName;
  mutable ::capnp::ParsedSchema rootSchema;
  mutable ::capnp::StructSchema typeSchema;
};
} // namespace detail
} // namespace capnp
} // namespace esi
} // namespace circt

/// Write a valid capnp schema to memory, then parse it out of memory using the
/// capnp library. Writing and parsing text within a single process is ugly, but
/// this is by far the easiest way to do this. This isn't the use case for which
/// Cap'nProto was designed.
::capnp::ParsedSchema TypeSchemaImpl::getSchema() const {
  if (rootSchema != ::capnp::ParsedSchema())
    return rootSchema;

  // Write the schema to `schemaText`.
  std::string schemaText;
  llvm::raw_string_ostream os(schemaText);
  emitId(os, 0xFFFFFFFFFFFFFFFF) << ";\n";
  auto rc = write(os);
  assert(succeeded(rc) && "Failed schema text output.");
  os.str();

  // Write `schemaText` to an in-memory filesystem then parse it. Yes, this is
  // the only way to do this.
  kj::Own<kj::Filesystem> fs = kj::newDiskFilesystem();
  kj::Own<kj::Directory> dir = kj::newInMemoryDirectory(kj::nullClock());
  kj::Path fakePath = kj::Path::parse("schema.capnp");
  { // Ensure that 'fakeFile' has flushed.
    auto fakeFile = dir->openFile(fakePath, kj::WriteMode::CREATE);
    fakeFile->writeAll(schemaText);
  }
  rootSchema = parser.parseFromDirectory(*dir, std::move(fakePath), nullptr);
  return rootSchema;
}

/// Find the schema corresponding to `type` and return it.
::capnp::StructSchema TypeSchemaImpl::getTypeSchema() const {
  if (typeSchema != ::capnp::StructSchema())
    return typeSchema;
  uint64_t id = capnpTypeID();
  for (auto schemaNode : getSchema().getAllNested()) {
    if (schemaNode.getProto().getId() == id) {
      typeSchema = schemaNode.asStruct();
      return typeSchema;
    }
  }
  assert(false && "A node with a matching ID should always be found.");
}

// We compute a deterministic hash based on the type. Since llvm::hash_value
// changes from execution to execution, we don't use it. This assumes a closed
// type system, which is reasonable since we only support some types in the
// Capnp schema generation anyway.
uint64_t TypeSchemaImpl::capnpTypeID() const {
  if (cachedID)
    return *cachedID;

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
  cachedID = hash | 0x8000000000000000;
  return *cachedID;
}

/// Returns true if the type is currently supported.
bool TypeSchemaImpl::isSupported() const {
  return llvm::TypeSwitch<::mlir::Type, bool>(type)
      .Case<IntegerType>([](IntegerType t) { return t.getWidth() <= 64; })
      .Default([](Type) { return false; });
}

// Compute the expected size of the capnp message in bits.
size_t TypeSchemaImpl::size() const {
  auto schema = getTypeSchema();
  auto structProto = schema.getProto().getStruct();
  return 64 * // Convert from 64-bit words to bits.
         (2 + // Headers
          structProto.getDataWordCount() + structProto.getPointerCount());
}

/// For now, the name is just the type serialized. This works only because we
/// only support ints.
StringRef TypeSchemaImpl::name() const {
  if (cachedName == "") {
    llvm::raw_string_ostream os(cachedName);
    os << "TY" << type;
    cachedName = os.str();
  }
  return cachedName;
}

/// This function is essentially a placeholder which only supports ints. It'll
/// need to be re-worked when we start supporting structs, arrays, unions,
/// enums, etc.
LogicalResult TypeSchemaImpl::write(llvm::raw_ostream &rawOS) const {
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

LogicalResult TypeSchemaImpl::writeMetadata(llvm::raw_ostream &os) const {
  os << name() << " ";
  emitId(os, capnpTypeID());
  return success();
}

bool TypeSchemaImpl::operator==(const TypeSchemaImpl &that) const {
  return type == that.type;
}

//===----------------------------------------------------------------------===//
// Capnp encode / decode RTL builders.
//
// These have the potential to get large and complex as we add more types. The
// encoding spec is here: https://capnproto.org/encoding.html
//===----------------------------------------------------------------------===//

static size_t bits(::capnp::schema::Type::Reader type) {
  using ty = ::capnp::schema::Type;
  switch (type.which()) {
  case ty::VOID:
    return 0;
  case ty::UINT8:
  case ty::INT8:
    return 8;
  case ty::UINT16:
  case ty::INT16:
    return 16;
  case ty::UINT32:
  case ty::INT32:
    return 32;
  case ty::UINT64:
  case ty::INT64:
    return 64;
  default:
    assert(false && "Type not yet supported");
  }
}

/// Build an RTL/SV dialect capnp encoder for this type.
Value TypeSchemaImpl::buildEncoder(OpBuilder &b, Value operand) {
  auto loc = operand.getDefiningOp()->getLoc();
  return b.create<rtl::ConstantOp>(loc, 0, type.cast<IntegerType>());
}

/// Build an RTL/SV dialect capnp decoder for this type.
Value TypeSchemaImpl::buildDecoder(OpBuilder &b, Value operand) {
  MLIRContext *ctxt = b.getContext();
  auto loc = operand.getDefiningOp()->getLoc();
  size_t size = this->size();

  // Various useful integer types.
  auto u16 = IntegerType::get(ctxt, 16);
  auto u32 = b.getI32Type();
  auto u64 = b.getI64Type();
  // Various useful bit array types.
  auto b16 = rtl::ArrayType::get(b.getI1Type(), 16);
  auto b32 = rtl::ArrayType::get(b.getI1Type(), 32);
  auto b64 = rtl::ArrayType::get(b.getI1Type(), 64);

  rtl::ArrayType operandType = operand.getType().dyn_cast<rtl::ArrayType>();
  assert(operandType && operandType.getSize() == size &&
         "Operand type and length must match the type's capnp size.");

  // capnp messages start with the segment table, which for a single segment is
  // just the size of the message minus the 64-bit segment table. Create an
  // assertion that it matches our computed size.
  auto segSize = b.create<rtl::ReinterpretCast>(
      loc, u64, b.create<rtl::ArraySliceOp>(loc, operand, size - 64, 64));
  auto expectedSize = b.create<rtl::ConstantOp>(loc, size - 64, u64);
  b.create<sv::AssertOp>(loc, b.create<rtl::ICmpOp>(loc, b.getI1Type(),
                                                    ICmpPredicate::eq, segSize,
                                                    expectedSize));
  size_t currentOffset = 64;

  // The next 64-bits of a capnp message is the root struct pointer.
  ::capnp::schema::Node::Reader rootProto = getTypeSchema().getProto();
  auto ptr =
      b.create<rtl::ArraySliceOp>(loc, operand, size - currentOffset - 64, 64);

  // Since this is the root, we _expect_ the offset to be zero but that's only
  // guaranteed to be the case with canonically-encoded messages.
  // TODO: support cases where the pointer offset is non-zero.
  auto typeAndOffset = b.create<rtl::ReinterpretCast>(
      loc, u32, b.create<rtl::ArraySliceOp>(loc, ptr, 32, 32));
  auto b16Zero = b.create<rtl::ConstantOp>(loc, 0, u32);
  b.create<sv::AssertOp>(loc, b.create<rtl::ICmpOp>(loc, b.getI1Type(),
                                                    ICmpPredicate::eq,
                                                    typeAndOffset, b16Zero));

  // We expect the data section to be equal to the computed data section size.
  auto dataSectionSize = b.create<rtl::ReinterpretCast>(
      loc, u16, b.create<rtl::ArraySliceOp>(loc, ptr, 16, 16));
  auto expectedDataSectionSize = b.create<rtl::ConstantOp>(
      loc, rootProto.getStruct().getDataWordCount() * 64, u16);
  b.create<sv::AssertOp>(
      loc, b.create<rtl::ICmpOp>(loc, b.getI1Type(), ICmpPredicate::eq,
                                 dataSectionSize, expectedDataSectionSize));

  // We expect the pointer section to be equal to the computed pointer section
  // size.
  auto ptrSectionSize = b.create<rtl::ReinterpretCast>(
      loc, u16, b.create<rtl::ArraySliceOp>(loc, ptr, 0, 16));
  auto expectedPtrSectionSize = b.create<rtl::ConstantOp>(
      loc, rootProto.getStruct().getPointerCount() * 64, u16);
  b.create<sv::AssertOp>(
      loc, b.create<rtl::ICmpOp>(loc, b.getI1Type(), ICmpPredicate::eq,
                                 ptrSectionSize, expectedPtrSectionSize));
  // Done looking at the ptr.
  currentOffset += 64;

  // Now that we're looking at the data section, we can just cast down each
  // type. Since we only support IntegerType, this is easy.
  auto field = rootProto.getStruct().getFields()[0];
  size_t capnpFieldBits = bits(field.getSlot().getType());
  currentOffset += (field.getSlot().getOffset() + 1) * capnpFieldBits;
  auto typeBits = type.cast<IntegerType>().getWidth();
  auto fieldBits =
      b.create<rtl::ArraySliceOp>(loc, operand, size - currentOffset, 1);
  auto fieldValue = b.create<rtl::ReinterpretCast>(loc, type, fieldBits);

  // All that just to decode an int!
  return fieldValue;
}

//===----------------------------------------------------------------------===//
// TypeSchema wrapper.
//===----------------------------------------------------------------------===//

circt::esi::capnp::TypeSchema::TypeSchema(Type type) {
  circt::esi::ChannelPort chan = type.dyn_cast<circt::esi::ChannelPort>();
  if (chan) // Unwrap the channel if it's a channel.
    type = chan.getInner();
  s = std::make_shared<detail::TypeSchemaImpl>(type);
}
Type circt::esi::capnp::TypeSchema::getType() const { return s->getType(); }
uint64_t circt::esi::capnp::TypeSchema::capnpTypeID() const {
  return s->capnpTypeID();
}
bool circt::esi::capnp::TypeSchema::isSupported() const {
  return s->isSupported();
}
size_t circt::esi::capnp::TypeSchema::size() const { return s->size(); }
StringRef circt::esi::capnp::TypeSchema::name() const { return s->name(); }
LogicalResult
circt::esi::capnp::TypeSchema::write(llvm::raw_ostream &os) const {
  return s->write(os);
}
LogicalResult
circt::esi::capnp::TypeSchema::writeMetadata(llvm::raw_ostream &os) const {
  return s->writeMetadata(os);
}
bool circt::esi::capnp::TypeSchema::operator==(const TypeSchema &that) const {
  return *s == *that.s;
}
Value circt::esi::capnp::TypeSchema::buildEncoder(OpBuilder &builder,
                                                  Value operand) const {
  return s->buildEncoder(builder, operand);
}
Value circt::esi::capnp::TypeSchema::buildDecoder(OpBuilder &builder,
                                                  Value operand) const {
  return s->buildDecoder(builder, operand);
}
