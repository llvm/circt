//===- FIRRTLTypes.cpp - Implement the FIRRTL dialect type system ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the FIRRTL dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

using mlir::TypeStorageAllocator;

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void FIRRTLType::print(raw_ostream &os) const {
  auto printWidthQualifier = [&](Optional<int32_t> width) {
    if (width)
      os << '<' << width.getValue() << '>';
  };

  TypeSwitch<FIRRTLType>(*this)
      .Case<ClockType>([&](Type) { os << "clock"; })
      .Case<ResetType>([&](Type) { os << "reset"; })
      .Case<AsyncResetType>([&](Type) { os << "asyncreset"; })
      .Case<SIntType>([&](SIntType sIntType) {
        os << "sint";
        printWidthQualifier(sIntType.getWidth());
      })
      .Case<UIntType>([&](UIntType uIntType) {
        os << "uint";
        printWidthQualifier(uIntType.getWidth());
      })
      .Case<AnalogType>([&](AnalogType analogType) {
        os << "analog";
        printWidthQualifier(analogType.getWidth());
      })
      .Case<FlipType>([&](FlipType flipType) {
        os << "flip<";
        flipType.getElementType().print(os);
        os << '>';
      })
      .Case<BundleType>([&](BundleType bundleType) {
        os << "bundle<";
        llvm::interleaveComma(bundleType.getElements(), os,
                              [&](BundleType::BundleElement element) {
                                os << element.name.getValue() << ": ";
                                element.type.print(os);
                              });
        os << '>';
      })
      .Case<FVectorType>([&](FVectorType vectorType) {
        os << "vector<";
        vectorType.getElementType().print(os);
        os << ", " << vectorType.getNumElements() << '>';
      })
      .Default([](Type) { assert(0 && "unknown dialect type to print"); });
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

/// type
///   ::= clock
///   ::= reset
///   ::= asyncreset
///   ::= sint ('<' int '>')?
///   ::= uint ('<' int '>')?
///   ::= analog ('<' int '>')?
///   ::= flip '<' type '>'
///   ::= bundle '<' (bundle-elt (',' bundle-elt)*)? '>'
///   ::= vector '<' type ',' int '>'
///
/// bundle-elt ::= identifier ':' type
///
static ParseResult parseType(FIRRTLType &result, DialectAsmParser &parser) {
  StringRef name;
  if (parser.parseKeyword(&name))
    return failure();

  auto *context = parser.getBuilder().getContext();

  if (name.equals("clock")) {
    return result = ClockType::get(context), success();
  } else if (name.equals("reset")) {
    return result = ResetType::get(context), success();
  } else if (name.equals("asyncreset")) {
    return result = AsyncResetType::get(context), success();
  } else if (name.equals("sint") || name.equals("uint") ||
             name.equals("analog")) {
    // Parse the width specifier if it exists.
    int32_t width = -1;
    if (!parser.parseOptionalLess()) {
      if (parser.parseInteger(width) || parser.parseGreater())
        return failure();

      if (width < 0)
        return parser.emitError(parser.getNameLoc(), "unknown width"),
               failure();
    }

    if (name.equals("sint"))
      result = SIntType::get(context, width);
    else if (name.equals("uint"))
      result = UIntType::get(context, width);
    else {
      assert(name.equals("analog"));
      result = AnalogType::get(context, width);
    }
    return success();
  } else if (name.equals("flip")) {
    FIRRTLType element;
    if (parser.parseLess() || parseType(element, parser) ||
        parser.parseGreater())
      return failure();
    return result = FlipType::get(element), success();
  } else if (name.equals("bundle")) {
    if (parser.parseLess())
      return failure();

    SmallVector<BundleType::BundleElement, 4> elements;
    if (parser.parseOptionalGreater()) {
      // Parse all of the bundle-elt's.
      do {
        std::string nameStr;
        StringRef name;
        FIRRTLType type;

        // The 'name' can be an identifier or an integer.
        auto parseIntOrStringName = [&]() -> ParseResult {
          uint32_t fieldIntName;
          auto intName = parser.parseOptionalInteger(fieldIntName);
          if (intName.hasValue()) {
            nameStr = llvm::utostr(fieldIntName);
            name = nameStr;
            return intName.getValue();
          }

          // Otherwise must be an identifier.
          return parser.parseKeyword(&name);
          return success();
        };

        if (parseIntOrStringName() || parser.parseColon() ||
            parseType(type, parser))
          return failure();

        elements.push_back({StringAttr::get(context, name), type});
      } while (!parser.parseOptionalComma());

      if (parser.parseGreater())
        return failure();
    }

    return result = BundleType::get(elements, context), success();
  } else if (name.equals("vector")) {
    FIRRTLType elementType;
    unsigned width = 0;

    if (parser.parseLess() || parseType(elementType, parser) ||
        parser.parseComma() || parser.parseInteger(width) ||
        parser.parseGreater())
      return failure();

    return result = FVectorType::get(elementType, width), success();
  }

  return parser.emitError(parser.getNameLoc(), "unknown firrtl type"),
         failure();
}

/// Parse a type registered to this dialect.
Type FIRRTLDialect::parseType(DialectAsmParser &parser) const {
  FIRRTLType result;
  if (::parseType(result, parser))
    return Type();
  return result;
}

//===----------------------------------------------------------------------===//
// FIRRTLType Implementation
//===----------------------------------------------------------------------===//

/// Return true if this is a 'ground' type, aka a non-aggregate type.
bool FIRRTLType::isGround() {
  return TypeSwitch<FIRRTLType, bool>(*this)
      .Case<ClockType, ResetType, AsyncResetType, SIntType, UIntType,
            AnalogType>([](Type) { return true; })
      .Case<BundleType, FVectorType>([](Type) { return false; })
      .Case<FlipType>(
          [](FlipType type) { return type.getElementType().isGround(); })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return false;
      });
}

/// Return a pair with the 'isPassive' and 'containsAnalog' bits.
std::pair<bool, bool> FIRRTLType::getRecursiveTypeProperties() {
  return TypeSwitch<FIRRTLType, std::pair<bool, bool>>(*this)
      .Case<ClockType, ResetType, AsyncResetType, SIntType, UIntType>(
          [](Type) { return std::make_pair(true, false); })
      .Case<AnalogType>([](Type) { return std::make_pair(true, true); })
      .Case<FlipType>([](Type) { return std::make_pair(false, false); })
      .Case<BundleType>([](BundleType bundleType) {
        return bundleType.getRecursiveTypeProperties();
      })
      .Case<FVectorType>([](FVectorType vectorType) {
        return vectorType.getRecursiveTypeProperties();
      })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return std::make_pair(false, false);
      });
}

/// Return this type with any flip types recursively removed from itself.
FIRRTLType FIRRTLType::getPassiveType() {
  return TypeSwitch<FIRRTLType, FIRRTLType>(*this)
      .Case<ClockType, ResetType, AsyncResetType, SIntType, UIntType,
            AnalogType>([&](Type) { return *this; })
      .Case<FlipType>([](FlipType flipType) {
        // Since types are _not_ canonicalized, a FlipType does _not_
        // guarantee that its elements are passive.
        return flipType.getElementType().getPassiveType();
      })
      .Case<BundleType>(
          [](BundleType bundleType) { return bundleType.getPassiveType(); })
      .Case<FVectorType>(
          [](FVectorType vectorType) { return vectorType.getPassiveType(); })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return FIRRTLType();
      });
}

/// Return this type with all ground types replaced with UInt<1>.  This is
/// used for `mem` operations.
FIRRTLType FIRRTLType::getMaskType() {
  return TypeSwitch<FIRRTLType, FIRRTLType>(*this)
      .Case<ClockType, ResetType, AsyncResetType, SIntType, UIntType,
            AnalogType>(
          [&](Type) { return UIntType::get(this->getContext(), 1); })
      .Case<FlipType>([](FlipType flipType) {
        return FlipType::get(flipType.getElementType().getMaskType());
      })
      .Case<BundleType>([&](BundleType bundleType) {
        SmallVector<BundleType::BundleElement, 4> newElements;
        newElements.reserve(bundleType.getElements().size());
        for (auto elt : bundleType.getElements())
          newElements.push_back({elt.name, elt.type.getMaskType()});
        return BundleType::get(newElements, this->getContext());
      })
      .Case<FVectorType>([](FVectorType vectorType) {
        return FVectorType::get(vectorType.getElementType().getMaskType(),
                                vectorType.getNumElements());
      })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return FIRRTLType();
      });
}

/// Remove the widths from this type. All widths are replaced with an
/// unknown width.
FIRRTLType FIRRTLType::getWidthlessType() {
  return TypeSwitch<FIRRTLType, FIRRTLType>(*this)
      .Case<ClockType, ResetType, AsyncResetType>([](auto a) { return a; })
      .Case<FlipType>([](FlipType a) {
        return FlipType::get(a.getElementType().getWidthlessType());
      })
      .Case<UIntType, SIntType, AnalogType>(
          [&](auto a) { return a.get(this->getContext(), -1); })
      .Case<BundleType>([&](auto a) {
        SmallVector<BundleType::BundleElement, 4> newElements;
        newElements.reserve(a.getElements().size());
        for (auto elt : a.getElements())
          newElements.push_back({elt.name, elt.type.getWidthlessType()});
        return BundleType::get(newElements, this->getContext());
      })
      .Case<FVectorType>([](auto a) {
        return FVectorType::get(a.getElementType().getWidthlessType(),
                                a.getNumElements());
      })
      .Default([](auto) {
        llvm_unreachable("unknown FIRRTL type");
        return FIRRTLType();
      });
}

/// If this is an IntType, AnalogType, or sugar type for a single bit (Clock,
/// Reset, etc) then return the bitwidth.  Return -1 if the is one of these
/// types but without a specified bitwidth.  Return -2 if this isn't a simple
/// type.
int32_t FIRRTLType::getBitWidthOrSentinel() {
  return TypeSwitch<FIRRTLType, int32_t>(*this)
      .Case<ClockType, ResetType, AsyncResetType>([](Type) { return 1; })
      .Case<SIntType, UIntType>(
          [&](IntType intType) { return intType.getWidthOrSentinel(); })
      .Case<AnalogType>(
          [](AnalogType analogType) { return analogType.getWidthOrSentinel(); })
      .Case<FlipType, BundleType, FVectorType>([](Type) { return -2; })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return -2;
      });
}

/// Return true if this is a type usable as a reset. This must be
/// either an abstract reset, a concrete 1-bit UInt, or an
/// asynchronous reset.
bool FIRRTLType::isResetType() {
  return TypeSwitch<FIRRTLType, bool>(*this)
      .Case<ResetType, AsyncResetType>([](Type) { return true; })
      .Case<UIntType>([](UIntType a) { return a.getWidth() == 1; })
      .Default([](Type) { return false; });
}

std::pair<FIRRTLType, bool> FIRRTLType::stripFlip() {
  if (auto a = this->dyn_cast<FlipType>())
    return {a.getElementType(), true};
  return {*this, false};
}

/// Helper to implement the equivalence logic for a pair of bundle elements.
/// Note that the FIRRTL spec requires bundle elements to have the same
/// orientation, but this only compares their passive types. The FIRRTL dialect
/// differs from the spec in how it uses flip types for module output ports and
/// canonicalizes flips in bundles, so only passive types can be compared here.
static bool areBundleElementsEquivalent(BundleType::BundleElement destElement,
                                        BundleType::BundleElement srcElement) {
  if (destElement.name != srcElement.name)
    return false;

  return areTypesEquivalent(destElement.type, srcElement.type);
}

/// Returns whether the two types are equivalent.  This implements the exact
/// definition of type equivalence in the FIRRTL spec.  If the types being
/// compared have any outer flips that encode FIRRTL module directions (input or
/// output), these should be stripped before using this method.
bool firrtl::areTypesEquivalent(FIRRTLType destType, FIRRTLType srcType) {
  // Reset types can be driven by UInt<1>, AsyncReset, or Reset types.
  if (destType.isa<ResetType>())
    return srcType.isResetType();

  // Reset types can drive UInt<1>, AsyncReset, or Reset types.
  if (srcType.isa<ResetType>())
    return destType.isResetType();

  // Vector types can be connected if they have the same size and element type.
  auto destVectorType = destType.dyn_cast<FVectorType>();
  auto srcVectorType = srcType.dyn_cast<FVectorType>();
  if (destVectorType && srcVectorType)
    return destVectorType.getNumElements() == srcVectorType.getNumElements() &&
           areTypesEquivalent(destVectorType.getElementType(),
                              srcVectorType.getElementType());

  // Bundle types can be connected if they have the same size, element names,
  // and element types.
  auto destBundleType = destType.dyn_cast<BundleType>();
  auto srcBundleType = srcType.dyn_cast<BundleType>();
  if (destBundleType && srcBundleType) {
    auto destElements = destBundleType.getElements();
    auto srcElements = srcBundleType.getElements();
    size_t numDestElements = destElements.size();
    if (numDestElements != srcElements.size())
      return false;

    for (size_t i = 0; i < numDestElements; ++i) {
      auto destElement = destElements[i];
      auto srcElement = srcElements[i];
      if (!areBundleElementsEquivalent(destElement, srcElement))
        return false;
    }
  }

  // Ground types can be connected if their passive, widthless versions
  // are equal.
  return destType.getWidthlessType() == srcType.getWidthlessType();
}

/// Returns whether the two types are weakly equivalent.
bool firrtl::areTypesWeaklyEquivalent(FIRRTLType destType, FIRRTLType srcType,
                                      bool destFlip, bool srcFlip) {

  // Reset types can be driven by UInt<1>, AsyncReset, or Reset types.
  if (destType.isa<ResetType>())
    return srcType.isResetType();

  // Reset types can drive UInt<1>, AsyncReset, or Reset types.
  if (srcType.isa<ResetType>())
    return destType.isResetType();

  // Vector types can be connected if their element types are weakly equivalent.
  // Size doesn't matter.
  auto destVectorType = destType.dyn_cast<FVectorType>();
  auto srcVectorType = srcType.dyn_cast<FVectorType>();
  if (destVectorType && srcVectorType)
    return areTypesWeaklyEquivalent(destVectorType.getElementType(),
                                    srcVectorType.getElementType(), destFlip,
                                    srcFlip);

  // Bundle types are weakly equivalent if all common elements are weakly
  // equivalent.  Non-matching fields are ignored.  Flips are "pushed" into
  // recursive weak type equivalence checks.
  auto destBundleType = destType.dyn_cast<BundleType>();
  auto srcBundleType = srcType.dyn_cast<BundleType>();
  if (destBundleType && srcBundleType)
    return llvm::all_of(
        destBundleType.getElements(), [&](auto destElt) -> bool {
          auto destField = destElt.name.getValue();
          auto srcElt = srcBundleType.getElement(destField);
          // If the src doesn't contain the destination's field, that's okay.
          if (!srcElt)
            return true;
          auto a = destElt.type.stripFlip();
          auto b = srcElt.getValue().type.stripFlip();
          return areTypesWeaklyEquivalent(a.first, b.first, destFlip ^ a.second,
                                          srcFlip ^ b.second);
        });

  // Ground types can be connected if their passive, widthless versions
  // are equal and leaf flippedness matches.
  return destType.getWidthlessType() == srcType.getWidthlessType() &&
         destFlip == srcFlip;
}

/// Return the element of an array type or null.  This strips flip types.
Type firrtl::getVectorElementType(Type array) {
  if (auto flip = array.dyn_cast<FlipType>())
    array = flip.getElementType();
  auto vectorType = array.dyn_cast<FVectorType>();
  if (!vectorType)
    return Type();
  return vectorType.getElementType();
}

/// Return the passiver version of a firrtl type
/// top level for ODS constraint usage
Type firrtl::getPassiveType(Type anyFIRRTLType) {
  return anyFIRRTLType.cast<FIRRTLType>().getPassiveType();
}

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

/// Return the bitwidth of this type or None if unknown.
Optional<int32_t> IntType::getWidth() {
  return isSigned() ? this->cast<SIntType>().getWidth()
                    : this->cast<UIntType>().getWidth();
}

/// Return a SIntType or UInt type with the specified signedness and width.
IntType IntType::get(MLIRContext *context, bool isSigned, int32_t width) {
  if (isSigned)
    return SIntType::get(context, width);
  return UIntType::get(context, width);
}

//===----------------------------------------------------------------------===//
// Width Qualified Ground Types
//===----------------------------------------------------------------------===//

namespace circt {
namespace firrtl {
namespace detail {
struct WidthTypeStorage : mlir::TypeStorage {
  WidthTypeStorage(int32_t width) : width(width) {}
  using KeyTy = int32_t;

  bool operator==(const KeyTy &key) const { return key == width; }

  static WidthTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<WidthTypeStorage>()) WidthTypeStorage(key);
  }

  int32_t width;
};
} // namespace detail
} // namespace firrtl
} // namespace circt

static Optional<int32_t>
getWidthQualifiedTypeWidth(firrtl::detail::WidthTypeStorage *impl) {
  int width = impl->width;
  if (width < 0)
    return None;
  return width;
}

/// Get an with a known width, or -1 for unknown.
SIntType SIntType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, width);
}

Optional<int32_t> SIntType::getWidth() {
  return getWidthQualifiedTypeWidth(this->getImpl());
}

UIntType UIntType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, width);
}

Optional<int32_t> UIntType::getWidth() {
  return getWidthQualifiedTypeWidth(this->getImpl());
}

/// Get an with a known width, or -1 for unknown.
AnalogType AnalogType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, width);
}

Optional<int32_t> AnalogType::getWidth() {
  return getWidthQualifiedTypeWidth(this->getImpl());
}

//===----------------------------------------------------------------------===//
// Flip Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace firrtl {
namespace detail {
struct FlipTypeStorage : mlir::TypeStorage {
  FlipTypeStorage(FIRRTLType element) : element(element) {}
  using KeyTy = FIRRTLType;

  bool operator==(const KeyTy &key) const { return key == element; }

  static FlipTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<FlipTypeStorage>()) FlipTypeStorage(key);
  }

  FIRRTLType element;
};

} // namespace detail
} // namespace firrtl
} // namespace circt

FIRRTLType FlipType::get(FIRRTLType element) {
  return TypeSwitch<FIRRTLType, FIRRTLType>(element)
      .Case<FlipType>([](auto flipType) {
        // flip(flip(x)) -> x
        return flipType.getElementType();
      })
      .Case<AnalogType>([](AnalogType analogType) {
        // flip(analog) -> analog.
        return analogType;
      })
      .Default([](FIRRTLType a) {
        return Base::get(a.getContext(), a).cast<FIRRTLType>();
      });
}

FIRRTLType FlipType::getElementType() { return getImpl()->element; }

//===----------------------------------------------------------------------===//
// Bundle Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace firrtl {
llvm::hash_code hash_value(const BundleType::BundleElement &arg) {
  return mlir::hash_value(arg.name) ^ mlir::hash_value(arg.type);
}
} // namespace firrtl
} // namespace circt

enum {
  /// Bit set if the type only contains passive elements.
  IsPassiveBitMask = 0x1,
  /// Bit set if the type contains an analog type.
  ContainsAnalogBitMask = 0x2,
};

namespace circt {
namespace firrtl {
namespace detail {
struct BundleTypeStorage : mlir::TypeStorage {
  using KeyTy = ArrayRef<BundleType::BundleElement>;

  BundleTypeStorage(KeyTy elements)
      : elements(elements.begin(), elements.end()) {
    bool isPassive = true, containsAnalog = false;
    unsigned fieldID = 0;
    fieldIDs.reserve(elements.size());
    for (auto &element : elements) {
      auto type = element.type;
      auto eltInfo = type.getRecursiveTypeProperties();
      isPassive &= eltInfo.first;
      containsAnalog |= eltInfo.second;
      fieldID += 1;
      // If the element is a bundle type, increment the field id by the number
      // of sub-fields, so that each field gets a unique number.
      if (auto bundleType = type.dyn_cast<BundleType>())
        fieldID += bundleType.getMaxFieldID();
      fieldIDs.push_back(fieldID);
    }
    maxFieldID = fieldID;
    unsigned flags = 0;
    if (isPassive)
      flags |= IsPassiveBitMask;
    if (containsAnalog)
      flags |= ContainsAnalogBitMask;
    passiveContainsAnalogTypeInfo.setInt(flags);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(elements); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  static BundleTypeStorage *construct(TypeStorageAllocator &allocator,
                                      KeyTy key) {
    return new (allocator.allocate<BundleTypeStorage>()) BundleTypeStorage(key);
  }

  SmallVector<BundleType::BundleElement, 4> elements;
  SmallVector<unsigned, 4> fieldIDs;
  unsigned maxFieldID;

  /// This holds two bits indicating whether the current type is passive and
  /// if it contains an analog type, and can hold a pointer to a passive type if
  /// not.
  llvm::PointerIntPair<Type, 2, unsigned> passiveContainsAnalogTypeInfo;
};

} // namespace detail
} // namespace firrtl
} // namespace circt

FIRRTLType BundleType::get(ArrayRef<BundleElement> elements,
                           MLIRContext *context) {
  return Base::get(context, elements);
}

auto BundleType::getElements() -> ArrayRef<BundleElement> {
  return getImpl()->elements;
}

/// Return a pair with the 'isPassive' and 'containsAnalog' bits.
std::pair<bool, bool> BundleType::getRecursiveTypeProperties() {
  auto flags = getImpl()->passiveContainsAnalogTypeInfo.getInt();
  return std::make_pair((flags & IsPassiveBitMask) != 0,
                        (flags & ContainsAnalogBitMask) != 0);
}

/// Return this type with any flip types recursively removed from itself.
FIRRTLType BundleType::getPassiveType() {
  auto *impl = getImpl();

  // If we've already determined and cached the passive type, use it.
  if (auto passiveType = impl->passiveContainsAnalogTypeInfo.getPointer())
    return passiveType.cast<FIRRTLType>();

  // If this type is already passive, use it and remember for next time.
  if (impl->passiveContainsAnalogTypeInfo.getInt() & IsPassiveBitMask) {
    impl->passiveContainsAnalogTypeInfo.setPointer(*this);
    return *this;
  }

  // Otherwise at least one element is non-passive, rebuild a passive version.
  SmallVector<BundleType::BundleElement, 16> newElements;
  newElements.reserve(impl->elements.size());
  for (auto &elt : impl->elements) {
    newElements.push_back({elt.name, elt.type.getPassiveType()});
  }

  auto passiveType = BundleType::get(newElements, getContext());
  impl->passiveContainsAnalogTypeInfo.setPointer(passiveType);
  return passiveType;
}

llvm::Optional<unsigned> BundleType::getElementIndex(StringRef name) {
  for (auto it : llvm::enumerate(getElements())) {
    auto element = it.value();
    if (element.name.getValue() == name) {
      return {it.index()};
    }
  }
  return None;
}

/// Look up an element by name.  This returns a BundleElement with.
auto BundleType::getElement(StringRef name) -> Optional<BundleElement> {
  if (auto maybeIndex = getElementIndex(name))
    return getElements()[*maybeIndex];
  return None;
}

FIRRTLType BundleType::getElementType(StringRef name) {
  auto element = getElement(name);
  return element.hasValue() ? element.getValue().type : FIRRTLType();
}

unsigned BundleType::getFieldID(unsigned index) {
  return getImpl()->fieldIDs[index];
}

unsigned BundleType::getIndexForFieldID(unsigned fieldID) {
  assert(getElements().size() && "Bundle must have >0 fields");
  auto fieldIDs = getImpl()->fieldIDs;
  auto it = std::prev(std::upper_bound(fieldIDs.begin(), fieldIDs.end(), fieldID));
  return std::distance(fieldIDs.begin(), it);
}

unsigned BundleType::getMaxFieldID() {
  return getImpl()->maxFieldID;
}

//===----------------------------------------------------------------------===//
// Vector Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace firrtl {
namespace detail {
struct VectorTypeStorage : mlir::TypeStorage {
  using KeyTy = std::pair<FIRRTLType, unsigned>;

  VectorTypeStorage(KeyTy value) : value(value) {
    auto properties = value.first.getRecursiveTypeProperties();
    unsigned flags = 0;
    if (properties.first)
      flags |= IsPassiveBitMask;
    if (properties.second)
      flags |= ContainsAnalogBitMask;
    passiveContainsAnalogTypeInfo.setInt(flags);
  }

  bool operator==(const KeyTy &key) const { return key == value; }

  static VectorTypeStorage *construct(TypeStorageAllocator &allocator,
                                      KeyTy key) {
    return new (allocator.allocate<VectorTypeStorage>()) VectorTypeStorage(key);
  }

  KeyTy value;

  /// This holds two bits indicating whether the current type is passive and
  /// if it contains an analog type, and can hold a pointer to a passive type if
  /// not.
  llvm::PointerIntPair<Type, 2, unsigned> passiveContainsAnalogTypeInfo;
};

} // namespace detail
} // namespace firrtl
} // namespace circt

FIRRTLType FVectorType::get(FIRRTLType elementType, unsigned numElements) {
  // If elementType is a flip, then we canonicalize it outwards.
  if (auto flip = elementType.dyn_cast<FlipType>())
    return FlipType::get(FVectorType::get(flip.getElementType(), numElements));

  return Base::get(elementType.getContext(),
                   std::make_pair(elementType, numElements));
}

FIRRTLType FVectorType::getElementType() { return getImpl()->value.first; }

unsigned FVectorType::getNumElements() { return getImpl()->value.second; }

/// Return a pair with the 'isPassive' and 'containsAnalog' bits.
std::pair<bool, bool> FVectorType::getRecursiveTypeProperties() {
  auto flags = getImpl()->passiveContainsAnalogTypeInfo.getInt();
  return std::make_pair((flags & IsPassiveBitMask) != 0,
                        (flags & ContainsAnalogBitMask) != 0);
}

/// Return this type with any flip types recursively removed from itself.
FIRRTLType FVectorType::getPassiveType() {
  auto *impl = getImpl();

  // If we've already determined and cached the passive type, use it.
  if (auto passiveType = impl->passiveContainsAnalogTypeInfo.getPointer())
    return passiveType.cast<FIRRTLType>();

  // If this type is already passive, return it and remember for next time.
  if (impl->passiveContainsAnalogTypeInfo.getInt() & IsPassiveBitMask) {
    impl->passiveContainsAnalogTypeInfo.setPointer(*this);
    return *this;
  }

  // Otherwise, rebuild a passive version.
  auto passiveType =
      FVectorType::get(getElementType().getPassiveType(), getNumElements());
  impl->passiveContainsAnalogTypeInfo.setPointer(passiveType);
  return passiveType;
}

void FIRRTLDialect::registerTypes() {
  addTypes<SIntType, UIntType, ClockType, ResetType, AsyncResetType, AnalogType,
           // Derived Types
           FlipType, BundleType, FVectorType>();
}
