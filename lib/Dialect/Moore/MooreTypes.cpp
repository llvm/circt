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
      .Case<RealType>([](auto type) { return 64; })
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
