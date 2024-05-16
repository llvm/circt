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
using mlir::DialectAsmParser;
using mlir::DialectAsmPrinter;
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
  addTypes<PackedUnsizedDim, PackedRangeDim, UnpackedUnsizedDim,
           UnpackedArrayDim, UnpackedRangeDim, UnpackedAssocDim,
           UnpackedQueueDim, PackedStructType, UnpackedStructType>();

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

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

Domain UnpackedType::getDomain() const {
  return TypeSwitch<UnpackedType, Domain>(*this)
      .Case<PackedType>([](auto type) { return type.getDomain(); })
      .Case<UnpackedDim>([&](auto type) { return type.getInner().getDomain(); })
      .Case<UnpackedStructType>(
          [](auto type) { return type.getStruct().domain; })
      .Default([](auto) { return Domain::TwoValued; });
}

std::optional<unsigned> UnpackedType::getBitSize() const {
  return TypeSwitch<UnpackedType, std::optional<unsigned>>(*this)
      .Case<PackedType>([](auto type) { return type.getBitSize(); })
      .Case<RealType>([](auto type) { return 64; })
      .Case<UnpackedUnsizedDim>([](auto) { return std::nullopt; })
      .Case<UnpackedArrayDim>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getInner().getBitSize())
          return (*size) * type.getSize();
        return {};
      })
      .Case<UnpackedRangeDim>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getInner().getBitSize())
          return (*size) * type.getRange().size;
        return {};
      })
      .Case<UnpackedStructType>(
          [](auto type) { return type.getStruct().bitSize; })
      .Default([](auto) { return std::nullopt; });
}

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

Domain PackedType::getDomain() const {
  return TypeSwitch<PackedType, Domain>(*this)
      .Case<VoidType>([](auto) { return Domain::TwoValued; })
      .Case<IntType>([&](auto type) { return type.getDomain(); })
      .Case<PackedDim>([&](auto type) { return type.getInner().getDomain(); })
      .Case<PackedStructType>(
          [](auto type) { return type.getStruct().domain; });
}

std::optional<unsigned> PackedType::getBitSize() const {
  return TypeSwitch<PackedType, std::optional<unsigned>>(*this)
      .Case<VoidType>([](auto) { return 0; })
      .Case<IntType>([](auto type) { return type.getWidth(); })
      .Case<PackedUnsizedDim>([](auto) { return std::nullopt; })
      .Case<PackedRangeDim>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getInner().getBitSize())
          return (*size) * type.getRange().size;
        return {};
      })
      .Case<PackedStructType>(
          [](auto type) { return type.getStruct().bitSize; });
}

//===----------------------------------------------------------------------===//
// Packed Dimensions
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {

struct DimStorage : TypeStorage {
  using KeyTy = UnpackedType;

  DimStorage(KeyTy key) : inner(key) {}
  bool operator==(const KeyTy &key) const { return key == inner; }
  static DimStorage *construct(TypeStorageAllocator &allocator,
                               const KeyTy &key) {
    return new (allocator.allocate<DimStorage>()) DimStorage(key);
  }

  UnpackedType inner;
};

struct UnsizedDimStorage : DimStorage {
  UnsizedDimStorage(KeyTy key) : DimStorage(key) {}
  static UnsizedDimStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<UnsizedDimStorage>()) UnsizedDimStorage(key);
  }
};

struct RangeDimStorage : DimStorage {
  using KeyTy = std::pair<UnpackedType, Range>;

  RangeDimStorage(KeyTy key) : DimStorage(key.first), range(key.second) {}
  bool operator==(const KeyTy &key) const {
    return key.first == inner && key.second == range;
  }
  static RangeDimStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<RangeDimStorage>()) RangeDimStorage(key);
  }

  Range range;
};

} // namespace detail
} // namespace moore
} // namespace circt

PackedType PackedDim::getInner() const {
  return llvm::cast<PackedType>(getImpl()->inner);
}

std::optional<Range> PackedDim::getRange() const {
  if (auto dim = llvm::dyn_cast<PackedRangeDim>(*this))
    return dim.getRange();
  return {};
}

std::optional<unsigned> PackedDim::getSize() const {
  return llvm::transformOptional(getRange(), [](auto r) { return r.size; });
}

const detail::DimStorage *PackedDim::getImpl() const {
  return static_cast<detail::DimStorage *>(this->impl);
}

PackedUnsizedDim PackedUnsizedDim::get(PackedType inner) {
  return Base::get(inner.getContext(), inner);
}

PackedRangeDim PackedRangeDim::get(PackedType inner, Range range) {
  return Base::get(inner.getContext(), inner, range);
}

Range PackedRangeDim::getRange() const { return getImpl()->range; }

//===----------------------------------------------------------------------===//
// Unpacked Dimensions
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {

struct SizedDimStorage : DimStorage {
  using KeyTy = std::pair<UnpackedType, unsigned>;

  SizedDimStorage(KeyTy key) : DimStorage(key.first), size(key.second) {}
  bool operator==(const KeyTy &key) const {
    return key.first == inner && key.second == size;
  }
  static SizedDimStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<SizedDimStorage>()) SizedDimStorage(key);
  }

  unsigned size;
};

struct AssocDimStorage : DimStorage {
  using KeyTy = std::pair<UnpackedType, UnpackedType>;

  AssocDimStorage(KeyTy key) : DimStorage(key.first), indexType(key.second) {}
  bool operator==(const KeyTy &key) const {
    return key.first == inner && key.second == indexType;
  }
  static AssocDimStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<AssocDimStorage>()) AssocDimStorage(key);
  }

  UnpackedType indexType;
};

} // namespace detail
} // namespace moore
} // namespace circt

UnpackedType UnpackedDim::getInner() const { return getImpl()->inner; }

const detail::DimStorage *UnpackedDim::getImpl() const {
  return static_cast<detail::DimStorage *>(this->impl);
}

UnpackedUnsizedDim UnpackedUnsizedDim::get(UnpackedType inner) {
  return Base::get(inner.getContext(), inner);
}

UnpackedArrayDim UnpackedArrayDim::get(UnpackedType inner, unsigned size) {
  return Base::get(inner.getContext(), inner, size);
}

unsigned UnpackedArrayDim::getSize() const { return getImpl()->size; }

UnpackedRangeDim UnpackedRangeDim::get(UnpackedType inner, Range range) {
  return Base::get(inner.getContext(), inner, range);
}

Range UnpackedRangeDim::getRange() const { return getImpl()->range; }

UnpackedAssocDim UnpackedAssocDim::get(UnpackedType inner,
                                       UnpackedType indexType) {
  return Base::get(inner.getContext(), inner, indexType);
}

UnpackedType UnpackedAssocDim::getIndexType() const {
  return getImpl()->indexType;
}

UnpackedQueueDim UnpackedQueueDim::get(UnpackedType inner,
                                       std::optional<unsigned> bound) {
  return Base::get(inner.getContext(), inner, bound.value_or(-1));
}

std::optional<unsigned> UnpackedQueueDim::getBound() const {
  unsigned bound = getImpl()->size;
  if (bound == static_cast<unsigned>(-1))
    return {};
  return bound;
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

struct Subset {
  enum { None, Unpacked, Packed } implied = None;
  bool allowUnpacked = true;
};

static ParseResult parseMooreType(DialectAsmParser &parser, Subset subset,
                                  Type &type);
static void printMooreType(Type type, DialectAsmPrinter &printer,
                           Subset subset);

/// Parse a type with custom syntax.
static OptionalParseResult customTypeParser(DialectAsmParser &parser,
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

  // Packed and unpacked ranges.
  if (mnemonic == "unsized") {
    UnpackedType inner;
    if (parser.parseLess() || parseMooreType(parser, subset, inner) ||
        parser.parseGreater())
      return failure();
    return yieldImplied(
        [&]() { return PackedUnsizedDim::get(cast<PackedType>(inner)); },
        [&]() { return UnpackedUnsizedDim::get(inner); });
  }
  if (mnemonic == "range") {
    UnpackedType inner;
    int left, right;
    if (parser.parseLess() || parseMooreType(parser, subset, inner) ||
        parser.parseComma() || parser.parseInteger(left) ||
        parser.parseColon() || parser.parseInteger(right) ||
        parser.parseGreater())
      return failure();
    return yieldImplied(
        [&]() {
          return PackedRangeDim::get(cast<PackedType>(inner), left, right);
        },
        [&]() { return UnpackedRangeDim::get(inner, left, right); });
  }
  if (mnemonic == "array") {
    UnpackedType inner;
    unsigned size;
    if (parser.parseLess() || parseMooreType(parser, subset, inner) ||
        parser.parseComma() || parser.parseInteger(size) ||
        parser.parseGreater())
      return failure();
    return yieldUnpacked(UnpackedArrayDim::get(inner, size));
  }
  if (mnemonic == "assoc") {
    UnpackedType inner;
    UnpackedType index;
    if (parser.parseLess() || parseMooreType(parser, subset, inner))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parseMooreType(parser, {Subset::Unpacked, true}, index))
        return failure();
    }
    if (parser.parseGreater())
      return failure();
    return yieldUnpacked(UnpackedAssocDim::get(inner, index));
  }
  if (mnemonic == "queue") {
    UnpackedType inner;
    std::optional<unsigned> size;
    if (parser.parseLess() || parseMooreType(parser, subset, inner))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      unsigned tmpSize;
      if (parser.parseInteger(tmpSize))
        return failure();
      size = tmpSize;
    }
    if (parser.parseGreater())
      return failure();
    return yieldUnpacked(UnpackedQueueDim::get(inner, size));
  }

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
static LogicalResult customTypePrinter(Type type, DialectAsmPrinter &printer,
                                       Subset subset) {
  // If we are printing a type that may be both packed or unpacked, emit a
  // wrapping `packed<...>` or `unpacked<...>` accordingly if not done so
  // previously, in order to disambiguate between the two.
  if (llvm::isa<PackedDim>(type) || llvm::isa<UnpackedDim>(type) ||
      llvm::isa<PackedStructType>(type) ||
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

      // Packed and unpacked dimensions
      .Case<PackedUnsizedDim, UnpackedUnsizedDim>([&](auto type) {
        printer << "unsized<";
        printMooreType(type.getInner(), printer, subset);
        printer << ">";
        return success();
      })
      .Case<PackedRangeDim, UnpackedRangeDim>([&](auto type) {
        printer << "range<";
        printMooreType(type.getInner(), printer, subset);
        printer << ", " << type.getRange() << ">";
        return success();
      })
      .Case<UnpackedArrayDim>([&](auto type) {
        printer << "array<";
        printMooreType(type.getInner(), printer, subset);
        printer << ", " << type.getSize() << ">";
        return success();
      })
      .Case<UnpackedAssocDim>([&](auto type) {
        printer << "assoc<";
        printMooreType(type.getInner(), printer, subset);
        if (auto indexType = type.getIndexType()) {
          printer << ", ";
          printMooreType(indexType, printer, {Subset::Unpacked, true});
        }
        printer << ">";
        return success();
      })
      .Case<UnpackedQueueDim>([&](auto type) {
        printer << "queue<";
        printMooreType(type.getInner(), printer, subset);
        if (auto bound = type.getBound())
          printer << ", " << *bound;
        printer << ">";
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
static ParseResult parseMooreType(DialectAsmParser &parser, Subset subset,
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
static void printMooreType(Type type, DialectAsmPrinter &printer,
                           Subset subset) {
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
