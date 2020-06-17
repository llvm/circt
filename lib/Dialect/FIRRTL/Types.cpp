//===- Types.cpp - Implement the FIRRTL dialect type system ---------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Types.h"
#include "circt/Dialect/FIRRTL/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

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

  switch (getKind()) {
  default:
    assert(0 && "unknown dialect type to print");
    // Ground Types Without Parameters
  case FIRRTLType::Clock:
    os << "clock";
    break;
  case FIRRTLType::Reset:
    os << "reset";
    break;
  case FIRRTLType::AsyncReset:
    os << "asyncreset";
    break;

  // Width Qualified Types
  case FIRRTLType::SInt:
    os << "sint";
    printWidthQualifier(cast<SIntType>().getWidth());
    break;
  case FIRRTLType::UInt:
    os << "uint";
    printWidthQualifier(cast<UIntType>().getWidth());
    break;
  case FIRRTLType::Analog:
    os << "analog";
    printWidthQualifier(cast<AnalogType>().getWidth());
    break;

    // Derived Types
  case FIRRTLType::Flip:
    os << "flip<";
    cast<FlipType>().getElementType().print(os);
    os << '>';
    break;

  case FIRRTLType::Bundle:
    os << "bundle<";
    llvm::interleaveComma(cast<BundleType>().getElements(), os,
                          [&](BundleType::BundleElement element) {
                            os << element.first << ": ";
                            element.second.print(os);
                          });
    os << '>';
    break;
  case FIRRTLType::Vector: {
    auto vec = cast<FVectorType>();
    os << "vector<";
    vec.getElementType().print(os);
    os << ", " << vec.getNumElements() << '>';
    break;
  }
  }
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

  auto kind = llvm::StringSwitch<FIRRTLType::Kind>(name)
                  .Case("clock", FIRRTLType::Clock)
                  .Case("reset", FIRRTLType::Reset)
                  .Case("asyncreset", FIRRTLType::AsyncReset)
                  .Case("sint", FIRRTLType::SInt)
                  .Case("uint", FIRRTLType::UInt)
                  .Case("analog", FIRRTLType::Analog)
                  .Case("flip", FIRRTLType::Flip)
                  .Case("bundle", FIRRTLType::Bundle)
                  .Case("vector", FIRRTLType::Vector)
                  .Default(FIRRTLType::Kind(FIRRTLType::LAST_KIND + 1));
  auto *context = parser.getBuilder().getContext();

  switch (kind) {
  case FIRRTLType::Clock:
    return result = ClockType::get(context), success();
  case FIRRTLType::Reset:
    return result = ResetType::get(context), success();
  case FIRRTLType::AsyncReset:
    return result = AsyncResetType::get(context), success();

  case FIRRTLType::SInt:
  case FIRRTLType::UInt:
  case FIRRTLType::Analog: {
    // Parse the width specifier if it exists.
    int32_t width = -1;
    if (!parser.parseOptionalLess()) {
      if (parser.parseInteger(width) || parser.parseGreater())
        return failure();

      if (width < 0)
        return parser.emitError(parser.getNameLoc(), "unknown width"),
               failure();
    }

    if (kind == FIRRTLType::SInt)
      result = SIntType::get(context, width);
    else if (kind == FIRRTLType::UInt)
      result = UIntType::get(context, width);
    else {
      assert(kind == FIRRTLType::Analog);
      result = AnalogType::get(context, width);
    }
    return success();
  }

  case FIRRTLType::Flip: {
    FIRRTLType element;
    if (parser.parseLess() || parseType(element, parser) ||
        parser.parseGreater())
      return failure();
    return result = FlipType::get(element), success();
  }
  case FIRRTLType::Bundle: {
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
  }
  case FIRRTLType::Vector: {
    FIRRTLType elementType;
    unsigned width = 0;

    if (parser.parseLess() || parseType(elementType, parser) ||
        parser.parseComma() || parser.parseInteger(width) ||
        parser.parseGreater())
      return failure();

    return result = FVectorType::get(elementType, width), success();
  }
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
  switch (getKind()) {
  case FIRRTLType::Clock:
  case FIRRTLType::Reset:
  case FIRRTLType::AsyncReset:
  case FIRRTLType::SInt:
  case FIRRTLType::UInt:
  case FIRRTLType::Analog:
    return true;

    // Derived Types
  case FIRRTLType::Flip:
    return false;
  case FIRRTLType::Bundle:
    return this->cast<BundleType>().isPassiveType();
  case FIRRTLType::Vector:
    return this->cast<FVectorType>().isPassiveType();
  }
  llvm_unreachable("unknown FIRRTL type");
}

/// Return this type with any flip types recursively removed from itself.
FIRRTLType FIRRTLType::getPassiveType() {
  switch (getKind()) {
  case FIRRTLType::Clock:
  case FIRRTLType::Reset:
  case FIRRTLType::AsyncReset:
  case FIRRTLType::SInt:
  case FIRRTLType::UInt:
  case FIRRTLType::Analog:
    return *this;

    // Derived Types
  case FIRRTLType::Flip:
    return cast<FlipType>().getElementType().getPassiveType();
  case FIRRTLType::Bundle:
    return cast<BundleType>().getPassiveType();
  case FIRRTLType::Vector:
    return cast<FVectorType>().getPassiveType();
  }
  llvm_unreachable("unknown FIRRTL type");
}

/// Return this type with all ground types replaced with UInt<1>.  This is
/// used for `mem` operations.
FIRRTLType FIRRTLType::getMaskType() {
  switch (getKind()) {
  case FIRRTLType::Clock:
  case FIRRTLType::Reset:
  case FIRRTLType::AsyncReset:
  case FIRRTLType::SInt:
  case FIRRTLType::UInt:
  case FIRRTLType::Analog:
    return UIntType::get(getContext(), 1);

    // Derived Types
  case FIRRTLType::Flip:
    return FlipType::get(cast<FlipType>().getElementType().getMaskType());
  case FIRRTLType::Bundle: {
    auto bundle = cast<BundleType>();
    SmallVector<BundleType::BundleElement, 4> newElements;
    newElements.reserve(bundle.getElements().size());
    for (auto elt : bundle.getElements())
      newElements.push_back({elt.first, elt.second.getMaskType()});
    return BundleType::get(newElements, getContext());
  }
  case FIRRTLType::Vector: {
    auto vector = cast<FVectorType>();
    return FVectorType::get(vector.getElementType().getMaskType(),
                            vector.getNumElements());
  }
  }
  llvm_unreachable("unknown FIRRTL type");
}

/// If this is an IntType, AnalogType, or sugar type for a single bit (Clock,
/// Reset, etc) then return the bitwidth.  Return -1 if the is one of these
/// types but without a specified bitwidth.  Return -2 if this isn't a simple
/// type.
int32_t FIRRTLType::getBitWidthOrSentinel() {
  switch (getKind()) {
  default:
    assert(0 && "unknown FIRRTL type");
  case FIRRTLType::Clock:
  case FIRRTLType::Reset:
  case FIRRTLType::AsyncReset:
    return 1;
  case FIRRTLType::SInt:
  case FIRRTLType::UInt:
    return cast<IntType>().getWidthOrSentinel();
  case FIRRTLType::Analog:
    return cast<AnalogType>().getWidthOrSentinel();
  case FIRRTLType::Flip:
  case FIRRTLType::Bundle:
  case FIRRTLType::Vector:
    return -2;
  }
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
  return Base::get(context, SInt, width);
}

Optional<int32_t> SIntType::getWidth() {
  return getWidthQualifiedTypeWidth(this->getImpl());
}

UIntType UIntType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, UInt, width);
}

Optional<int32_t> UIntType::getWidth() {
  return getWidthQualifiedTypeWidth(this->getImpl());
}

/// Get an with a known width, or -1 for unknown.
AnalogType AnalogType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, Analog, width);
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
  switch (element.getKind()) {
  case FIRRTLType::Clock:
  case FIRRTLType::Reset:
  case FIRRTLType::AsyncReset:
  case FIRRTLType::SInt:
  case FIRRTLType::UInt:
  case FIRRTLType::Analog:
    break;

    // Derived Types
  case FIRRTLType::Flip:
    // flip(flip(x)) -> x
    return element.cast<FlipType>().getElementType();

  case FIRRTLType::Bundle: {
    // If the bundle is passive, then we're done because the flip will be at the
    // outer level. Otherwise, it contains flip types recursively within itself
    // that we should canonicalize.
    auto bundle = element.cast<BundleType>();
    if (bundle.isPassiveType())
      break;
    return getFlippedBundleType(bundle.getElements());
  }
  case FIRRTLType::Vector: {
    // If the bundle is passive, then we're done because the flip will be at the
    // outer level. Otherwise, it contains flip types recursively within itself
    // that we should canonicalize.
    auto vec = element.cast<FVectorType>();
    if (vec.isPassiveType())
      break;
    return FVectorType::get(get(vec.getElementType()), vec.getNumElements());
  }
  }

  // TODO: This should maintain a canonical form, digging any flips out of
  // sub-types.

  auto *context = element.getContext();
  return Base::get(context, Flip, element);
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

  return Base::get(context, Bundle, elements);
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

  return Base::get(elementType.getContext(), Vector,
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
