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
  TypeSchemaImpl(Type type);
  TypeSchemaImpl(const TypeSchemaImpl &) = delete;

  Type getType() const { return type; }

  uint64_t capnpTypeID() const;

  bool isSupported() const;
  size_t size() const;
  StringRef name() const;
  LogicalResult write(llvm::raw_ostream &os) const;
  LogicalResult writeMetadata(llvm::raw_ostream &os) const;

  bool operator==(const TypeSchemaImpl &) const;

  /// Build an RTL/SV dialect capnp encoder for this type.
  Value buildEncoder(OpBuilder &, Value clk, Value valid, Value);
  /// Build an RTL/SV dialect capnp decoder for this type.
  Value buildDecoder(OpBuilder &, Value clk, Value valid, Value);

private:
  ::capnp::ParsedSchema getSchema() const;
  ::capnp::StructSchema getTypeSchema() const;

  Type type;
  /// Capnp requires that everything be contained in a struct. ESI doesn't so we
  /// wrap non-struct types in a capnp struct. During decoder/encoder
  /// construction, it's convenient to use the capnp model so assemble the
  /// virtual list of `Type`s here.
  ArrayRef<Type> fieldTypes;

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

TypeSchemaImpl::TypeSchemaImpl(Type t) : type(t) {
  fieldTypes =
      TypeSwitch<Type, ArrayRef<Type>>(type)
          .Case([this](IntegerType) { return ArrayRef<Type>(&type, 1); })
          .Default([](Type) { return ArrayRef<Type>(); });
}

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
// changes from execution to execution, we don't use it.
uint64_t TypeSchemaImpl::capnpTypeID() const {
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

/// Returns true if the type is currently supported.
static bool isSupported(Type type) {
  return llvm::TypeSwitch<::mlir::Type, bool>(type)
      .Case<IntegerType>([](IntegerType t) { return t.getWidth() <= 64; })
      .Case<rtl::ArrayType>(
          [](rtl::ArrayType t) { return isSupported(t.getElementType()); })
      .Default([](Type) { return false; });
}

/// Returns true if the type is currently supported.
bool TypeSchemaImpl::isSupported() const { return ::isSupported(type); }

// Compute the expected size of the capnp message in bits.
size_t TypeSchemaImpl::size() const {
  auto schema = getTypeSchema();
  auto structProto = schema.getProto().getStruct();
  return 64 * // Convert from 64-bit words to bits.
         (1 + // Header
          structProto.getDataWordCount() + structProto.getPointerCount());
}

/// Write a valid Capnp name for 'type'.
static void emitName(Type type, llvm::raw_ostream &os) {
  llvm::TypeSwitch<Type>(type)
      .Case([&os](IntegerType intTy) {
        std::string intName;
        llvm::raw_string_ostream(intName) << intTy;
        // Capnp struct names must start with an uppercase character.
        intName[0] = toupper(intName[0]);
        os << intName;
      })
      .Case([&os](rtl::ArrayType arrTy) {
        os << "ArrayOf";
        emitName(arrTy.getElementType(), os);
      })
      .Default([](Type) {
        assert(false && "Type not supported. Please check support first with "
                        "isSupported()");
      });
}

/// For now, the name is just the type serialized. This works only because we
/// only support ints.
StringRef TypeSchemaImpl::name() const {
  if (cachedName == "") {
    llvm::raw_string_ostream os(cachedName);
    emitName(type, os);
    cachedName = os.str();
  }
  return cachedName;
}

/// Write a valid Capnp type.
static void emitCapnpType(Type type, IndentingOStream &os) {
  llvm::TypeSwitch<Type>(type)
      .Case([&os](IntegerType intTy) {
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
            assert(false && "Type not supported. Integer too wide. Please "
                            "check support first with isSupported()");
        }
      })
      .Case([&os](rtl::ArrayType arrTy) {
        os << "List(";
        emitCapnpType(arrTy.getElementType(), os);
        os << ')';
      })
      .Default([](Type) {
        assert(false && "Type not supported. Please check support first with "
                        "isSupported()");
      });
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

  // Specify the actual type, followed by the capnp field.
  os.indent() << "# Actual type is " << type << ".\n";
  os.indent() << "i @0 :";
  emitCapnpType(type, os);
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

/// Build an RTL/SV dialect capnp encoder for this type. Inputs need to be
/// packed on unpadded.
Value TypeSchemaImpl::buildEncoder(OpBuilder &b, Value clk, Value valid,
                                   Value operand) {
  MLIRContext *ctxt = b.getContext();
  auto loc = operand.getDefiningOp()->getLoc();
  ::capnp::schema::Node::Reader rootProto = getTypeSchema().getProto();

  auto i16 = b.getIntegerType(16);
  auto i32 = b.getIntegerType(32);

  auto typeAndOffset = b.create<rtl::ConstantOp>(loc, i32, 0);
  auto ptrSize = b.create<rtl::ConstantOp>(loc, i16, 0);
  auto dataSize = b.create<rtl::ConstantOp>(
      loc, i16, rootProto.getStruct().getDataWordCount());
  auto structPtr = b.create<rtl::ConcatOp>(
      loc, ValueRange{ptrSize, dataSize, typeAndOffset});

  auto operandIntTy = operand.getType().cast<IntegerType>();
  uint16_t paddingBits =
      rootProto.getStruct().getDataWordCount() * 64 - operandIntTy.getWidth();
  auto operandCasted = b.create<rtl::BitcastOp>(
      loc,
      IntegerType::get(ctxt, operandIntTy.getWidth(), IntegerType::Signless),
      operand);

  IntegerType iPaddingTy = IntegerType::get(ctxt, paddingBits);
  auto padding = b.create<rtl::ConstantOp>(loc, iPaddingTy, 0);
  auto dataSection =
      b.create<rtl::ConcatOp>(loc, ValueRange{padding, operandCasted});

  return b.create<rtl::ConcatOp>(loc, ValueRange{dataSection, structPtr});
}

namespace {
/// Holds a 'slice' of an array and is able to construct more slice ops, then
/// cast to a type.
struct Slice {
public:
  Slice(OpBuilder &b, Value init) : builder(&b), s(init) {
    assert(init.getType().isa<rtl::ArrayType>());
  }

  /// Create an op to slice the array from lsb to lsb + size. Return a new slice
  /// with that op.
  Slice slice(int64_t lsb, int64_t size) {
    Value newSlice = builder->create<rtl::ArraySliceOp>(loc(), s, lsb, size);
    return Slice(*builder, newSlice);
  }

  /// Set the "name" attribute of a slice's op.
  Slice name(const Twine &name) const {
    SmallString<32> nameStr;
    s.getDefiningOp()->setAttr(
        "name", StringAttr::get(name.toStringRef(nameStr), ctxt()));
    return *this;
  }
  Slice name(capnp::Text::Reader fieldName, const Twine &nameSuffix) const {
    return name(StringRef(fieldName.cStr()) + nameSuffix);
  }

  /// Construct a bitcast.
  Value cast(Type t, StringRef name = StringRef(), Twine nameSuffix = Twine()) {
    auto dst = builder->create<rtl::BitcastOp>(loc(), t, s);

    SmallString<32> nameBuffer;
    StringRef wholeName = (name + nameSuffix).toStringRef(nameBuffer);
    if (wholeName.size())
      dst->setAttr("name", StringAttr::get(wholeName, ctxt()));
    return dst;
  }

  Operation *operator->() const { return s.getDefiningOp(); }
  Value getValue() const { return s; }
  Location loc() const { return s.getLoc(); }
  OpBuilder &b() const { return *builder; }
  MLIRContext *ctxt() const { return builder->getContext(); }

private:
  OpBuilder *builder;
  Value s;
};
} // namespace

/// Construct the proper operations to convert a capnp field to 'type'.
static Value decodeField(Type type, capnp::schema::Field::Reader field,
                         Slice dataSection, Slice ptrSection,
                         OpBuilder &asserts) {
  Slice fieldSlice = TypeSwitch<Type, Slice>(type).Case([&](IntegerType it) {
    return dataSection.slice(field.getSlot().getOffset() *
                                 bits(field.getSlot().getType()),
                             it.getWidth());
  });

  fieldSlice.name(field.getName(), "_bits");
  return fieldSlice.cast(type, field.getName().cStr(), "Value");
}

/// Build an RTL/SV dialect capnp decoder for this type. Outputs packed and
/// unpadded data.
Value TypeSchemaImpl::buildDecoder(OpBuilder &b, Value clk, Value valid,
                                   Value operandVal) {
  // Various useful integer types.
  auto i16 = b.getIntegerType(16);
  auto i32 = b.getIntegerType(32);

  size_t size = this->size();
  rtl::ArrayType operandType = operandVal.getType().dyn_cast<rtl::ArrayType>();
  assert(operandType && operandType.getSize() == size &&
         "Operand type and length must match the type's capnp size.");

  Slice operand(b, operandVal);
  auto loc = operand.loc();

  auto alwaysAt = b.create<sv::AlwaysOp>(loc, EventControl::AtPosEdge, clk);
  auto ifValid =
      OpBuilder(alwaysAt.getBodyRegion()).create<sv::IfOp>(loc, valid);
  OpBuilder asserts(ifValid.getBodyRegion());

  // The next 64-bits of a capnp message is the root struct pointer.
  ::capnp::schema::Node::Reader rootProto = getTypeSchema().getProto();
  auto ptr = operand.slice(0, 64).name("rootPointer");

  // Since this is the root, we _expect_ the offset to be zero but that's only
  // guaranteed to be the case with canonically-encoded messages.
  // TODO: support cases where the pointer offset is non-zero.
  Slice assertPtr(ptr);
  auto typeAndOffset = assertPtr.slice(0, 32).cast(i32, "typeAndOffset");
  auto b16Zero = asserts.create<rtl::ConstantOp>(loc, i32, 0);

  asserts.create<sv::AssertOp>(
      loc, asserts.create<rtl::ICmpOp>(loc, b.getI1Type(), ICmpPredicate::eq,
                                       typeAndOffset, b16Zero));

  // We expect the data section to be equal to the computed data section size.
  auto dataSectionSize = assertPtr.slice(32, 16).cast(i16, "dataSectionSize");
  auto expectedDataSectionSize = asserts.create<rtl::ConstantOp>(
      loc, i16, rootProto.getStruct().getDataWordCount());
  asserts.create<sv::AssertOp>(
      loc,
      asserts.create<rtl::ICmpOp>(loc, b.getI1Type(), ICmpPredicate::eq,
                                  dataSectionSize, expectedDataSectionSize));

  // We expect the pointer section to be equal to the computed pointer section
  // size.
  auto ptrSectionSize = assertPtr.slice(48, 16).cast(i16, "ptrSectionSize");
  auto expectedPtrSectionSize = asserts.create<rtl::ConstantOp>(
      loc, i16, rootProto.getStruct().getPointerCount() * 64);
  asserts.create<sv::AssertOp>(
      loc, asserts.create<rtl::ICmpOp>(loc, b.getI1Type(), ICmpPredicate::eq,
                                       ptrSectionSize, expectedPtrSectionSize));

  // Get pointers to the data and pointer sections.
  auto st = rootProto.getStruct();
  auto dataSection =
      operand.slice(64, st.getDataWordCount() * 64).name("dataSection");
  auto ptrSection = operand
                        .slice(64 + (st.getDataWordCount() * 64),
                               rootProto.getStruct().getPointerCount() * 64)
                        .name("ptrSection");

  // Loop through fields.
  Value fieldValues[st.getFields().size()];
  for (auto field : st.getFields()) {
    uint16_t idx = field.getCodeOrder();
    assert(idx < fieldTypes.size() && "Capnp struct longer than fieldTypes.");
    fieldValues[idx] =
        decodeField(fieldTypes[idx], field, dataSection, ptrSection, asserts);
  }

  // What to return depends on the type. (e.g. structs have to be constructed
  // from the field values.)
  return TypeSwitch<Type, Value>(type).Case(
      [&fieldValues](IntegerType) { return fieldValues[0]; });
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
Value circt::esi::capnp::TypeSchema::buildEncoder(OpBuilder &builder, Value clk,
                                                  Value valid,
                                                  Value operand) const {
  return s->buildEncoder(builder, clk, valid, operand);
}
Value circt::esi::capnp::TypeSchema::buildDecoder(OpBuilder &builder, Value clk,
                                                  Value valid,
                                                  Value operand) const {
  return s->buildDecoder(builder, clk, valid, operand);
}
