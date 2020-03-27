//===- Types.cpp - Implement the FIRRTL dialect type system ---------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"
#include "llvm/ADT/StringSwitch.h"

using namespace spt;
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

  case FIRRTLType::Bundle: {
    os << "bundle<";
    mlir::interleaveComma(cast<BundleType>().getElements(), os,
                          [&](BundleType::BundleElement element) {
                            os << element.first << ": ";
                            element.second.print(os);
                          });
    os << ">";
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
///   ::= sint ('<' int '>')?
///   ::= uint ('<' int '>')?
///   ::= analog ('<' int '>')?
///   ::= flip '<' type '>'
///   ::= bundle '<' (bundle-elt (',' bundle-elt)*)? '>'
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
                  .Case("sint", FIRRTLType::SInt)
                  .Case("uint", FIRRTLType::UInt)
                  .Case("analog", FIRRTLType::Analog)
                  .Case("flip", FIRRTLType::Flip)
                  .Case("bundle", FIRRTLType::Bundle)
                  .Case("vector", FIRRTLType::Vector)
                  .Default(FIRRTLType::LAST_KIND);
  auto *context = parser.getBuilder().getContext();

  switch (kind) {
  case FIRRTLType::Clock:
    return result = ClockType::get(context), success();
  case FIRRTLType::Reset:
    return result = ResetType::get(context), success();

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
        StringRef name;
        FIRRTLType type;
        if (parser.parseKeyword(&name) || parser.parseColon() ||
            parseType(type, parser))
          return failure();

        elements.push_back({Identifier::get(name, context), type});
      } while (!parser.parseOptionalComma());

      if (parser.parseGreater())
        return failure();
    }

    return result = BundleType::get(elements, context), success();
  }
  case FIRRTLType::Vector:
  default:
    return parser.emitError(parser.getNameLoc(), "unknown firrtl type"),
           failure();
  }
}

/// Parse a type registered to this dialect.
Type FIRRTLDialect::parseType(DialectAsmParser &parser) const {
  FIRRTLType result;
  if (::parseType(result, parser))
    return Type();
  return result;
}

//===----------------------------------------------------------------------===//
// Width Qualified Ground Types
//===----------------------------------------------------------------------===//

namespace spt {
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

Optional<int32_t> getWidthQualifiedTypeWidth(WidthTypeStorage *impl) {
  int width = impl->width;
  if (width < 0)
    return None;
  return width;
}

} // namespace detail
} // namespace firrtl
} // namespace spt

/// Get an with a known width, or -1 for unknown.
SIntType SIntType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, SInt, width);
}

UIntType UIntType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, UInt, width);
}

/// Get an with a known width, or -1 for unknown.
AnalogType AnalogType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, Analog, width);
}

//===----------------------------------------------------------------------===//
// Flip Type
//===----------------------------------------------------------------------===//

namespace spt {
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
} // namespace spt

FlipType FlipType::get(FIRRTLType element) {
  auto *context = element.getContext();
  return Base::get(context, Flip, element);
}

FIRRTLType FlipType::getElementType() const { return getImpl()->element; }

//===----------------------------------------------------------------------===//
// Bundle Type
//===----------------------------------------------------------------------===//

namespace spt {
namespace firrtl {
namespace detail {
struct BundleTypeStorage : mlir::TypeStorage {
  using KeyTy = ArrayRef<BundleType::BundleElement>;

  BundleTypeStorage(KeyTy elements)
      : elements(elements.begin(), elements.end()) {}

  bool operator==(const KeyTy &key) const { return key == KeyTy(elements); }

  static BundleTypeStorage *construct(TypeStorageAllocator &allocator,
                                      KeyTy key) {
    return new (allocator.allocate<BundleTypeStorage>()) BundleTypeStorage(key);
  }

  SmallVector<BundleType::BundleElement, 4> elements;
};

} // namespace detail
} // namespace firrtl
} // namespace spt

BundleType BundleType::get(ArrayRef<BundleElement> elements,
                           MLIRContext *context) {
  return Base::get(context, Bundle, elements);
}

auto BundleType::getElements() const -> ArrayRef<BundleElement> {
  return getImpl()->elements;
}

//===----------------------------------------------------------------------===//
// Vector Type
//===----------------------------------------------------------------------===//

// FIXME: TODO
