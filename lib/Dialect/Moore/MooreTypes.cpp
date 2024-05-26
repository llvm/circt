//===- MooreTypes.cpp - Implement the Moore types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Moore dialect type system.
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
using mlir::LocationAttr;
using mlir::OptionalParseResult;
using mlir::StringSwitch;
using mlir::TypeStorage;
using mlir::TypeStorageAllocator;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.cpp.inc"

void MooreDialect::registerTypes() {
  addTypes<PackedStructType, UnpackedStructType>();

  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Moore/MooreTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

StringRef moore::getKeywordFromSign(const Sign &sign) {
  switch (sign) {
  case Sign::Unsigned:
    return "unsigned";
  case Sign::Signed:
    return "signed";
  }
  llvm_unreachable("all signs should be handled");
}

struct Subset {
  enum { None, Unpacked, Packed } implied = None;
  bool allowUnpacked = true;
};

static ParseResult parseMooreType(AsmParser &parser, Subset subset, Type &type);
static void printMooreType(Type type, AsmPrinter &printer, Subset subset);

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

Domain UnpackedType::getDomain() const {
  return TypeSwitch<UnpackedType, Domain>(*this)
      .Case<PackedType>([](auto type) { return type.getDomain(); })
      .Case<UnpackedArrayType, OpenUnpackedArrayType, AssocArrayType,
            QueueType>(
          [&](auto type) { return type.getElementType().getDomain(); })
      .Case<UnpackedStructType>(
          [](auto type) { return type.getStruct().domain; })
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
      .Case<UnpackedStructType>(
          [](auto type) { return type.getStruct().bitSize; })
      .Default([](auto) { return std::nullopt; });
}

Type UnpackedType::parse(mlir::AsmParser &parser) {
  Type type;
  if (parseMooreType(parser, {Subset::None, true}, type))
    return {};
  return type;
}

void UnpackedType::print(mlir::AsmPrinter &printer) const {
  printMooreType(*this, printer, {Subset::None, true});
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
      .Case<PackedStructType>(
          [](auto type) { return type.getStruct().domain; });
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
      .Case<PackedStructType>(
          [](auto type) { return type.getStruct().bitSize; });
}

//===----------------------------------------------------------------------===//
// Packed and Unpacked Structs
//===----------------------------------------------------------------------===//

StringRef moore::getMnemonicFromStructKind(StructKind kind) {
  switch (kind) {
  case StructKind::Struct:
    return "struct";
  case StructKind::Union:
    return "union";
  case StructKind::TaggedUnion:
    return "tagged_union";
  }
  llvm_unreachable("all struct kinds should be handled");
}

std::optional<StructKind> moore::getStructKindFromMnemonic(StringRef mnemonic) {
  return StringSwitch<std::optional<StructKind>>(mnemonic)
      .Case("struct", StructKind::Struct)
      .Case("union", StructKind::Union)
      .Case("tagged_union", StructKind::TaggedUnion)
      .Default({});
}

Struct::Struct(StructKind kind, ArrayRef<StructMember> members)
    : kind(kind), members(members.begin(), members.end()) {
  // The struct's value domain is two-valued if all members are two-valued.
  // Otherwise it is four-valued.
  domain = llvm::all_of(members,
                        [](auto &member) {
                          return member.type.getDomain() == Domain::TwoValued;
                        })
               ? Domain::TwoValued
               : Domain::FourValued;

  // The bit size is the sum of all member bit sizes, or `None` if any of the
  // member bit sizes are `None`.
  bitSize = 0;
  for (const auto &member : members) {
    if (auto memberSize = member.type.getBitSize()) {
      *bitSize += *memberSize;
    } else {
      bitSize = std::nullopt;
      break;
    }
  }
}

namespace circt {
namespace moore {
namespace detail {

struct StructTypeStorage : TypeStorage {
  using KeyTy = std::tuple<unsigned, ArrayRef<StructMember>>;

  StructTypeStorage(KeyTy key)
      : strukt(static_cast<StructKind>(std::get<0>(key)), std::get<1>(key)) {}
  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == static_cast<unsigned>(strukt.kind) &&
           std::get<1>(key) == ArrayRef<StructMember>(strukt.members);
  }
  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(key);
  }

  Struct strukt;
};

} // namespace detail
} // namespace moore
} // namespace circt

PackedStructType PackedStructType::get(MLIRContext *context, StructKind kind,
                                       ArrayRef<StructMember> members) {
  assert(llvm::all_of(members,
                      [](const StructMember &member) {
                        return llvm::isa<PackedType>(member.type);
                      }) &&
         "packed struct members must be packed");
  return Base::get(context, static_cast<unsigned>(kind), members);
}

const Struct &PackedStructType::getStruct() const { return getImpl()->strukt; }

UnpackedStructType UnpackedStructType::get(MLIRContext *context,
                                           StructKind kind,
                                           ArrayRef<StructMember> members) {
  return Base::get(context, static_cast<unsigned>(kind), members);
}

const Struct &UnpackedStructType::getStruct() const {
  return getImpl()->strukt;
}

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

/// Parse a type with custom syntax.
static OptionalParseResult customTypeParser(AsmParser &parser,
                                            StringRef mnemonic, Subset subset,
                                            llvm::SMLoc loc, Type &type) {
  auto *context = parser.getContext();

  auto yieldPacked = [&](PackedType x) {
    type = x;
    return success();
  };
  auto yieldUnpacked = [&](UnpackedType x) {
    if (!subset.allowUnpacked) {
      parser.emitError(loc)
          << "unpacked type " << x << " where only packed types are allowed";
      return failure();
    }
    type = x;
    return success();
  };
  auto yieldImplied =
      [&](llvm::function_ref<PackedType()> ifPacked,
          llvm::function_ref<UnpackedType()> ifUnpacked) {
        if (subset.implied == Subset::Packed)
          return yieldPacked(ifPacked());
        if (subset.implied == Subset::Unpacked)
          return yieldUnpacked(ifUnpacked());
        parser.emitError(loc)
            << "ambiguous packing; wrap `" << mnemonic
            << "` in `packed<...>` or `unpacked<...>` to disambiguate";
        return failure();
      };

  // Explicit packing indicators, like `unpacked.named`.
  if (mnemonic == "unpacked") {
    UnpackedType inner;
    if (parser.parseLess() ||
        parseMooreType(parser, {Subset::Unpacked, true}, inner) ||
        parser.parseGreater())
      return failure();
    return yieldUnpacked(inner);
  }
  if (mnemonic == "packed") {
    PackedType inner;
    if (parser.parseLess() ||
        parseMooreType(parser, {Subset::Packed, false}, inner) ||
        parser.parseGreater())
      return failure();
    return yieldPacked(inner);
  }

  // Packed primary types.
  if (mnemonic.size() > 1 && (mnemonic[0] == 'i' || mnemonic[0] == 'l') &&
      isdigit(mnemonic[1])) {
    auto domain = mnemonic[0] == 'i' ? Domain::TwoValued : Domain::FourValued;
    auto spelling = mnemonic.drop_front(1);
    unsigned width;
    if (spelling.getAsInteger(10, width))
      return parser.emitError(loc, "integer width invalid");
    return yieldPacked(IntType::get(context, width, domain));
  }

  // Everything that follows can be packed or unpacked. The packing is inferred
  // from the last `packed<...>` or `unpacked<...>` that we've seen. The
  // `yieldImplied` function will call the first lambda to construct a packed
  // type, or the second lambda to construct an unpacked type. If the
  // `subset.implied` field is not set, which means there hasn't been any prior
  // `packed` or `unpacked`, the function will emit an error properly.

  // Structs
  if (auto kind = getStructKindFromMnemonic(mnemonic)) {
    if (parser.parseLess())
      return failure();

    StringRef keyword;
    SmallVector<StructMember> members;
    auto result2 =
        parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Braces, [&]() {
          if (parser.parseKeyword(&keyword))
            return failure();
          UnpackedType type;
          if (parser.parseColon() || parseMooreType(parser, subset, type))
            return failure();
          members.push_back(
              {StringAttr::get(parser.getContext(), keyword), type});
          return success();
        });
    if (result2)
      return failure();

    return yieldImplied(
        [&]() {
          return PackedStructType::get(parser.getContext(), *kind, members);
        },
        [&]() {
          return UnpackedStructType::get(parser.getContext(), *kind, members);
        });
  }

  return {};
}

/// Print a type with custom syntax.
static LogicalResult customTypePrinter(Type type, AsmPrinter &printer,
                                       Subset subset) {
  // If we are printing a type that may be both packed or unpacked, emit a
  // wrapping `packed<...>` or `unpacked<...>` accordingly if not done so
  // previously, in order to disambiguate between the two.
  if (llvm::isa<PackedStructType>(type) ||
      llvm::isa<UnpackedStructType>(type)) {
    auto needed =
        llvm::isa<PackedType>(type) ? Subset::Packed : Subset::Unpacked;
    if (needed != subset.implied) {
      printer << (needed == Subset::Packed ? "packed" : "unpacked") << "<";
      printMooreType(type, printer, {needed, true});
      printer << ">";
      return success();
    }
  }

  return TypeSwitch<Type, LogicalResult>(type)
      // Integers and reals
      .Case<IntType>([&](auto type) {
        printer << (type.getDomain() == Domain::TwoValued ? "i" : "l");
        printer << type.getWidth();
        return success();
      })

      // Structs
      .Case<PackedStructType, UnpackedStructType>([&](auto type) {
        const auto &strukt = type.getStruct();
        printer << getMnemonicFromStructKind(strukt.kind) << "<{";
        llvm::interleaveComma(strukt.members, printer, [&](const auto &member) {
          printer << member.name.getValue() << ": ";
          printMooreType(member.type, printer, subset);
        });
        printer << "}>";
        return success();
      })

      .Default([](auto) { return failure(); });
}

/// Parse a type registered with this dialect.
static ParseResult parseMooreType(AsmParser &parser, Subset subset,
                                  Type &type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  StringRef mnemonic;
  if (auto result = generatedTypeParser(parser, &mnemonic, type);
      result.has_value())
    return result.value();

  if (auto result = customTypeParser(parser, mnemonic, subset, loc, type);
      result.has_value())
    return result.value();

  parser.emitError(loc) << "unknown type `" << mnemonic
                        << "` in dialect `moore`";
  return failure();
}

/// Print a type registered with this dialect.
static void printMooreType(Type type, AsmPrinter &printer, Subset subset) {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  if (succeeded(customTypePrinter(type, printer, subset)))
    return;
  assert(false && "no printer for unknown `moore` dialect type");
}

/// Parse a type registered with this dialect.
Type MooreDialect::parseType(DialectAsmParser &parser) const {
  Type type;
  if (parseMooreType(parser, {Subset::None, true}, type))
    return {};
  return type;
}

/// Print a type registered with this dialect.
void MooreDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printMooreType(type, printer, {Subset::None, true});
}
