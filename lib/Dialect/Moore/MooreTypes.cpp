//===- MooreTypes.cpp - Implement the Moore types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreTypes.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::moore;
using mlir::AsmParser;
using mlir::AsmPrinter;

bool moore::isIntType(Type type, unsigned width) {
  if (auto intType = dyn_cast<IntType>(type))
    return intType.getWidth() == width;
  return false;
}

bool moore::isIntType(Type type, Domain domain) {
  if (auto intType = dyn_cast<IntType>(type))
    return intType.getDomain() == domain;
  return false;
}

bool moore::isIntType(Type type, unsigned width, Domain domain) {
  if (auto intType = dyn_cast<IntType>(type))
    return intType.getWidth() == width && intType.getDomain() == domain;
  return false;
}

bool moore::isRealType(Type type, unsigned width) {
  if (auto realType = dyn_cast<RealType>(type))
    if (realType.getWidth() == RealWidth::f32)
      return width == 32;
  return width == 64;
}

static LogicalResult parseMembers(AsmParser &parser,
                                  SmallVector<StructLikeMember> &members);
static void printMembers(AsmPrinter &printer,
                         ArrayRef<StructLikeMember> members);

static ParseResult parseMooreType(AsmParser &parser, Type &type);
static void printMooreType(Type type, AsmPrinter &printer);

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

Domain UnpackedType::getDomain() const {
  return TypeSwitch<UnpackedType, Domain>(*this)
      .Case<PackedType>([](auto type) { return type.getDomain(); })
      .Case<UnpackedArrayType, OpenUnpackedArrayType, AssocArrayType,
            QueueType>(
          [&](auto type) { return type.getElementType().getDomain(); })
      .Case<UnpackedStructType, UnpackedUnionType>([](auto type) {
        for (const auto &member : type.getMembers())
          if (member.type.getDomain() == Domain::FourValued)
            return Domain::FourValued;
        return Domain::TwoValued;
      })
      .Default([](auto) { return Domain::TwoValued; });
}

std::optional<unsigned> UnpackedType::getBitSize() const {
  return TypeSwitch<UnpackedType, std::optional<unsigned>>(*this)
      .Case<PackedType>([](auto type) { return type.getBitSize(); })
      .Case<RealType>(
          [](auto type) { return type.getWidth() == RealWidth::f32 ? 32 : 64; })
      .Case<UnpackedArrayType>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getElementType().getBitSize())
          return (*size) * type.getSize();
        return {};
      })
      .Case<UnpackedStructType>([](auto type) -> std::optional<unsigned> {
        unsigned size = 0;
        for (const auto &member : type.getMembers()) {
          if (auto memberSize = member.type.getBitSize()) {
            size += *memberSize;
          } else {
            return std::nullopt;
          }
        }
        return size;
      })
      .Case<UnpackedUnionType>([](auto type) -> std::optional<unsigned> {
        unsigned size = 0;
        for (const auto &member : type.getMembers()) {
          if (auto memberSize = member.type.getBitSize()) {
            size = (*memberSize > size) ? *memberSize : size;
          } else {
            return std::nullopt;
          }
        }
        return size;
      })
      .Default([](auto) { return std::nullopt; });
}

Type UnpackedType::parse(mlir::AsmParser &parser) {
  Type type;
  if (parseMooreType(parser, type))
    return {};
  return type;
}

void UnpackedType::print(mlir::AsmPrinter &printer) const {
  printMooreType(*this, printer);
}

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

Domain PackedType::getDomain() const {
  return TypeSwitch<PackedType, Domain>(*this)
      .Case<VoidType>([](auto) { return Domain::TwoValued; })
      .Case<IntType>([&](auto type) { return type.getDomain(); })
      .Case<TimeType>([](auto) { return Domain::FourValued; })
      .Case<ArrayType, OpenArrayType>(
          [&](auto type) { return type.getElementType().getDomain(); })
      .Case<StructType, UnionType>([](auto type) {
        for (const auto &member : type.getMembers())
          if (member.type.getDomain() == Domain::FourValued)
            return Domain::FourValued;
        return Domain::TwoValued;
      });
}

std::optional<unsigned> PackedType::getBitSize() const {
  return TypeSwitch<PackedType, std::optional<unsigned>>(*this)
      .Case<VoidType>([](auto) { return 0; })
      .Case<IntType>([](auto type) { return type.getWidth(); })
      .Case<TimeType>([](auto type) { return 64; })
      .Case<ArrayType>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getElementType().getBitSize())
          return (*size) * type.getSize();
        return {};
      })
      .Case<OpenArrayType>([](auto) { return std::nullopt; })
      .Case<StructType>([](auto type) -> std::optional<unsigned> {
        unsigned size = 0;
        for (const auto &member : type.getMembers()) {
          if (auto memberSize = member.type.getBitSize()) {
            size += *memberSize;
          } else {
            return std::nullopt;
          }
        }
        return size;
      })
      .Case<UnionType>([](auto type) -> std::optional<unsigned> {
        unsigned size = 0;
        for (const auto &member : type.getMembers()) {
          if (auto memberSize = member.type.getBitSize()) {
            size = (*memberSize > size) ? *memberSize : size;
          } else {
            return std::nullopt;
          }
        }
        return size;
      })
      .Default([](auto) { return std::nullopt; });
}

IntType PackedType::getSimpleBitVector() const {
  if (auto intType = dyn_cast<IntType>(*this))
    return intType;
  if (auto bitSize = getBitSize())
    return IntType::get(getContext(), *bitSize, getDomain());
  return {};
}

bool PackedType::containsTimeType() const {
  return TypeSwitch<PackedType, bool>(*this)
      .Case<VoidType, IntType>([](auto) { return false; })
      .Case<TimeType>([](auto) { return true; })
      .Case<ArrayType, OpenArrayType>(
          [](auto type) { return type.getElementType().containsTimeType(); })
      .Case<StructType, UnionType>([](auto type) {
        return llvm::any_of(type.getMembers(), [](auto &member) {
          return cast<PackedType>(member.type).containsTimeType();
        });
      });
}

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

/// Parse a list of struct members.
static LogicalResult parseMembers(AsmParser &parser,
                                  SmallVector<StructLikeMember> &members) {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces, [&]() {
    std::string name;
    UnpackedType type;
    if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
        parser.parseCustomTypeWithFallback(type))
      return failure();

    members.push_back({StringAttr::get(parser.getContext(), name), type});
    return success();
  });
}

/// Print a list of struct members.
static void printMembers(AsmPrinter &printer,
                         ArrayRef<StructLikeMember> members) {
  printer << "{";
  llvm::interleaveComma(members, printer.getStream(),
                        [&](const StructLikeMember &member) {
                          printer.printKeywordOrString(member.name);
                          printer << ": ";
                          printer.printStrippedAttrOrType(member.type);
                        });
  printer << "}";
}

static LogicalResult
verifyAllMembersPacked(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<StructLikeMember> members) {
  if (!llvm::all_of(members, [](const auto &member) {
        return llvm::isa<PackedType>(member.type);
      }))
    return emitError() << "StructType/UnionType members must be packed types";
  return success();
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<StructLikeMember> members) {
  return verifyAllMembersPacked(emitError, members);
}

LogicalResult UnionType::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<StructLikeMember> members) {
  return verifyAllMembersPacked(emitError, members);
}

//===----------------------------------------------------------------------===//
// Interfaces for destructurable
//===----------------------------------------------------------------------===//

static std::optional<DenseMap<Attribute, Type>>
getAllSubelementIndexMap(ArrayRef<StructLikeMember> members) {
  DenseMap<Attribute, Type> destructured;
  for (const auto &member : members)
    destructured.insert({member.name, RefType::get(member.type)});
  return destructured;
}

static Type getTypeAtAllIndex(ArrayRef<StructLikeMember> members,
                              Attribute index) {
  auto indexAttr = cast<StringAttr>(index);
  if (!indexAttr)
    return {};
  for (const auto &member : members) {
    if (member.name == indexAttr) {
      return RefType::get(member.type);
    }
  }
  return Type();
}

static std::optional<uint32_t>
getFieldAllIndex(ArrayRef<StructLikeMember> members, StringAttr nameField) {
  for (uint32_t fieldIndex = 0; fieldIndex < members.size(); fieldIndex++)
    if (members[fieldIndex].name == nameField)
      return fieldIndex;
  return std::nullopt;
}

std::optional<DenseMap<Attribute, Type>> StructType::getSubelementIndexMap() {
  return getAllSubelementIndexMap(getMembers());
}

Type StructType::getTypeAtIndex(Attribute index) {
  return getTypeAtAllIndex(getMembers(), index);
}

std::optional<uint32_t> StructType::getFieldIndex(StringAttr nameField) {
  return getFieldAllIndex(getMembers(), nameField);
}

std::optional<DenseMap<Attribute, Type>>
UnpackedStructType::getSubelementIndexMap() {
  return getAllSubelementIndexMap(getMembers());
}

Type UnpackedStructType::getTypeAtIndex(Attribute index) {
  return getTypeAtAllIndex(getMembers(), index);
}

std::optional<uint32_t>
UnpackedStructType::getFieldIndex(StringAttr nameField) {
  return getFieldAllIndex(getMembers(), nameField);
}

std::optional<DenseMap<Attribute, Type>> UnionType::getSubelementIndexMap() {
  return getAllSubelementIndexMap(getMembers());
}

Type UnionType::getTypeAtIndex(Attribute index) {
  return getTypeAtAllIndex(getMembers(), index);
}

std::optional<uint32_t> UnionType::getFieldIndex(StringAttr nameField) {
  return getFieldAllIndex(getMembers(), nameField);
}

std::optional<DenseMap<Attribute, Type>>
UnpackedUnionType::getSubelementIndexMap() {
  return getAllSubelementIndexMap(getMembers());
}

Type UnpackedUnionType::getTypeAtIndex(Attribute index) {
  return getTypeAtAllIndex(getMembers(), index);
}

std::optional<uint32_t> UnpackedUnionType::getFieldIndex(StringAttr nameField) {
  return getFieldAllIndex(getMembers(), nameField);
}

std::optional<DenseMap<Attribute, Type>> RefType::getSubelementIndexMap() {
  return TypeSwitch<Type, std::optional<DenseMap<Attribute, Type>>>(
             getNestedType())
      .Case<StructType, UnpackedStructType>([](auto &type) {
        return getAllSubelementIndexMap(type.getMembers());
      })
      .Default([](auto) { return std::nullopt; });
}

Type RefType::getTypeAtIndex(Attribute index) {
  return TypeSwitch<Type, Type>(getNestedType())
      .Case<StructType, UnpackedStructType>([&index](auto &type) {
        return getTypeAtAllIndex(type.getMembers(), index);
      })
      .Default([](auto) { return Type(); });
}

std::optional<uint32_t> RefType::getFieldIndex(StringAttr nameField) {
  return TypeSwitch<Type, std::optional<uint32_t>>(getNestedType())
      .Case<StructType, UnpackedStructType>([&nameField](auto &type) {
        return getFieldAllIndex(type.getMembers(), nameField);
      })
      .Default([](auto) { return std::nullopt; });
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.cpp.inc"

void MooreDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Moore/MooreTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Parsing and printing
//===----------------------------------------------------------------------===//

/// Parse a type registered with this dialect.
static ParseResult parseMooreType(AsmParser &parser, Type &type) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  // Try the generated parser first.
  StringRef mnemonic;
  if (auto result = generatedTypeParser(parser, &mnemonic, type);
      result.has_value())
    return result.value();

  // Handle abbreviated integer types such as `i42` and `l42`.
  if (mnemonic.size() > 1 && (mnemonic[0] == 'i' || mnemonic[0] == 'l') &&
      isdigit(mnemonic[1])) {
    auto domain = mnemonic[0] == 'i' ? Domain::TwoValued : Domain::FourValued;
    auto spelling = mnemonic.drop_front(1);
    unsigned width;
    if (spelling.getAsInteger(10, width))
      return parser.emitError(loc, "integer width invalid");
    type = IntType::get(parser.getContext(), width, domain);
    return success();
  }

  if (mnemonic == "f32") {
    type = moore::RealType::get(parser.getContext(), /*width=*/RealWidth::f32);
    return success();
  }

  if (mnemonic == "f64") {
    type = moore::RealType::get(parser.getContext(), /*width=*/RealWidth::f64);
    return success();
  }

  parser.emitError(loc) << "unknown type `" << mnemonic
                        << "` in dialect `moore`";
  return failure();
}

/// Print a type registered with this dialect.
static void printMooreType(Type type, AsmPrinter &printer) {
  // Handle abbreviated integer types such as `i42` and `l42`.
  if (auto intType = dyn_cast<IntType>(type)) {
    printer << (intType.getDomain() == Domain::TwoValued ? "i" : "l");
    printer << intType.getWidth();
    return;
  }
  if (auto rt = dyn_cast<moore::RealType>(type)) {
    if (rt.getWidth() == RealWidth::f32) {
      printer << "f32";
      return;
    }
    printer << "f64";
    return;
  }

  // Otherwise fall back to the generated printer.
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  assert(false && "no printer for unknown `moore` dialect type");
}

/// Parse a type registered with this dialect.
Type MooreDialect::parseType(DialectAsmParser &parser) const {
  Type type;
  if (parseMooreType(parser, type))
    return {};
  return type;
}

/// Print a type registered with this dialect.
void MooreDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printMooreType(type, printer);
}
