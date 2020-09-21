//===- Types.cpp - Implement the FIRRTL dialect type system ---------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Types.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

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
                                os << element.first << ": ";
                                element.second.print(os);
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

        elements.push_back({Identifier::get(name, context), type});
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

/// Return true if this is a "passive" type - one that contains no "flip"
/// types recursively within itself.
bool FIRRTLType::isPassiveType() {
  return TypeSwitch<FIRRTLType, bool>(*this)
      .Case<ClockType, ResetType, AsyncResetType, SIntType, UIntType,
            AnalogType>([](Type) { return true; })
      .Case<FlipType>([](Type) { return false; })
      .Case<BundleType>(
          [](BundleType bundleType) { return bundleType.isPassiveType(); })
      .Case<FVectorType>(
          [](FVectorType vectorType) { return vectorType.isPassiveType(); })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return false;
      });
}

/// Return this type with any flip types recursively removed from itself.
FIRRTLType FIRRTLType::getPassiveType() {
  return TypeSwitch<FIRRTLType, FIRRTLType>(*this)
      .Case<ClockType, ResetType, AsyncResetType, SIntType, UIntType,
            AnalogType>([&](Type) { return *this; })
      .Case<FlipType>([](FlipType flipType) {
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
            AnalogType>([&](Type) { return UIntType::get(getContext(), 1); })
      .Case<FlipType>([](FlipType flipType) {
        return FlipType::get(flipType.getElementType().getMaskType());
      })
      .Case<BundleType>([&](BundleType bundleType) {
        SmallVector<BundleType::BundleElement, 4> newElements;
        newElements.reserve(bundleType.getElements().size());
        for (auto elt : bundleType.getElements())
          newElements.push_back({elt.first, elt.second.getMaskType()});
        return BundleType::get(newElements, getContext());
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
    .Case<ResetType, AsyncResetType>([](Type) {
      return true;
    })
    .Case<UIntType>([](UIntType a) {
      return a.getWidth() == 1;
    })
    .Default([](Type) {
      return false;
    });
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

/// Return a bundle type with the specified elements all flipped.  This assumes
/// the elements list is non-empty.
static FIRRTLType
getFlippedBundleType(ArrayRef<BundleType::BundleElement> elements) {
  assert(!elements.empty());
  SmallVector<BundleType::BundleElement, 16> flippedelements;
  flippedelements.reserve(elements.size());
  for (auto &elt : elements)
    flippedelements.push_back({elt.first, FlipType::get(elt.second)});
  return BundleType::get(flippedelements, elements[0].second.getContext());
}

FIRRTLType FlipType::get(FIRRTLType element) {
  // We maintain a canonical form for flip types, where we prefer to have the
  // flip as far outward from an otherwise passive type as possible.  If a flip
  // is being used with an aggregate type that contains non-passive elements,
  // then it is forced into the elements to get the canonical form.
  return TypeSwitch<FIRRTLType, FIRRTLType>(element)
      .Case<ClockType, ResetType, AsyncResetType, SIntType, UIntType,
            AnalogType>([&](Type) {
        // TODO: This should maintain a canonical form, digging any flips out of
        // sub-types.
        auto *context = element.getContext();
        return Base::get(context, element);
      })
      .Case<FlipType>([](FlipType flipType) {
        // flip(flip(x)) -> x
        return flipType.getElementType();
      })
      .Case<BundleType>([&](BundleType bundleType) {
        // If the bundle is passive, then we're done because the flip will be at
        // the outer level. Otherwise, it contains flip types recursively within
        // itself that we should canonicalize.
        if (bundleType.isPassiveType()) {
          auto *context = element.getContext();
          return Base::get(context, element).cast<FIRRTLType>();
        }

        return getFlippedBundleType(bundleType.getElements());
      })
      .Case<FVectorType>([&](FVectorType vectorType) {
        // If the bundle is passive, then we're done because the flip will be at
        // the outer level. Otherwise, it contains flip types recursively within
        // itself that we should canonicalize.
        if (vectorType.isPassiveType()) {
          auto *context = element.getContext();
          return Base::get(context, element).cast<FIRRTLType>();
        }

        return FVectorType::get(get(vectorType.getElementType()),
                                vectorType.getNumElements())
            .cast<FIRRTLType>();
      })
      .Default([](Type) {
        llvm_unreachable("unknown FIRRTL type");
        return FIRRTLType();
      });
}

FIRRTLType FlipType::getElementType() { return getImpl()->element; }

//===----------------------------------------------------------------------===//
// Bundle Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace firrtl {
namespace detail {
struct BundleTypeStorage : mlir::TypeStorage {
  using KeyTy = ArrayRef<BundleType::BundleElement>;

  BundleTypeStorage(KeyTy elements)
      : elements(elements.begin(), elements.end()) {

    bool isPassive = llvm::all_of(
        elements, [](const BundleType::BundleElement &elt) -> bool {
          auto eltType = elt.second;
          return eltType.isPassiveType();
        });
    passiveTypeInfo.setInt(isPassive);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(elements); }

  static BundleTypeStorage *construct(TypeStorageAllocator &allocator,
                                      KeyTy key) {
    return new (allocator.allocate<BundleTypeStorage>()) BundleTypeStorage(key);
  }

  SmallVector<BundleType::BundleElement, 4> elements;

  /// This holds a bit indicating whether the current type is passive, and
  /// can hold a pointer to a passive type if not.
  llvm::PointerIntPair<Type, 1, bool> passiveTypeInfo;
};

} // namespace detail
} // namespace firrtl
} // namespace circt

FIRRTLType BundleType::get(ArrayRef<BundleElement> elements,
                           MLIRContext *context) {
  // If all of the elements are flip types, then we canonicalize the flips to
  // the outer level.
  if (!elements.empty() &&
      llvm::all_of(elements, [&](const BundleElement &elt) -> bool {
        return elt.second.isa<FlipType>();
      })) {
    return FlipType::get(getFlippedBundleType(elements));
  }

  return Base::get(context, elements);
}

auto BundleType::getElements() -> ArrayRef<BundleElement> {
  return getImpl()->elements;
}

bool BundleType::isPassiveType() { return getImpl()->passiveTypeInfo.getInt(); }

/// Return this type with any flip types recursively removed from itself.
FIRRTLType BundleType::getPassiveType() {
  auto *impl = getImpl();
  // If this type is already passive, just return it.
  if (impl->passiveTypeInfo.getInt())
    return *this;

  // If we've already determined and cached the passive type, use it.
  if (auto passiveType = impl->passiveTypeInfo.getPointer())
    return passiveType.cast<FIRRTLType>();

  // Otherwise at least one element is non-passive, rebuild a passive version.
  SmallVector<BundleType::BundleElement, 16> newElements;
  newElements.reserve(impl->elements.size());
  for (auto &elt : impl->elements) {
    newElements.push_back({elt.first, elt.second.getPassiveType()});
  }

  auto passiveType = BundleType::get(newElements, getContext());
  impl->passiveTypeInfo.setPointer(passiveType);
  return passiveType;
}

/// Look up an element by name.  This returns a BundleElement with.
auto BundleType::getElement(StringRef name) -> Optional<BundleElement> {
  for (const auto &element : getElements()) {
    if (element.first == name)
      return element;
  }
  return None;
}

FIRRTLType BundleType::getElementType(StringRef name) {
  auto element = getElement(name);
  return element.hasValue() ? element.getValue().second : FIRRTLType();
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
    passiveTypeInfo.setInt(value.first.isPassiveType());
  }

  bool operator==(const KeyTy &key) const { return key == value; }

  static VectorTypeStorage *construct(TypeStorageAllocator &allocator,
                                      KeyTy key) {
    return new (allocator.allocate<VectorTypeStorage>()) VectorTypeStorage(key);
  }

  KeyTy value;

  /// This holds a bit indicating whether the current type is passive, and
  /// can hold a pointer to a passive type if not.
  llvm::PointerIntPair<Type, 1, bool> passiveTypeInfo;
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

bool FVectorType::isPassiveType() {
  return getImpl()->passiveTypeInfo.getInt();
}

/// Return this type with any flip types recursively removed from itself.
FIRRTLType FVectorType::getPassiveType() {
  auto *impl = getImpl();
  // If this type is already passive, just return it.
  if (impl->passiveTypeInfo.getInt())
    return *this;

  // If we've already determined and cached the passive type, use it.
  if (auto passiveType = impl->passiveTypeInfo.getPointer())
    return passiveType.cast<FIRRTLType>();

  // Otherwise, rebuild a passive version.
  auto passiveType =
      FVectorType::get(getElementType().getPassiveType(), getNumElements());
  impl->passiveTypeInfo.setPointer(passiveType);
  return passiveType;
}
