//===- Types.cpp - Implement the FIRRTL dialect type system ---------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "spt/Dialect/FIRRTL/IR/Ops.h"

using namespace spt;
using namespace firrtl;

/// Parse a type registered to this dialect.
Type FIRRTLDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeStr = parser.getFullSymbolSpec();

  if (typeStr == "clock")
    return ClockType::get(getContext());
  if (typeStr == "reset")
    return ResetType::get(getContext());

  // Parse the current typeStr as an optional width specifier like "<8>".  If
  // the string is empty, this returns -1.  If it is an integer it returns
  // the value.  If it is invalid, it emits an error and returns -2.
  auto parseWidth = [&]() -> int32_t {
    if (typeStr.empty())
      return -1;

    if (typeStr.front() != '<' || typeStr.back() != '>')
      return parser.emitError(parser.getNameLoc(), "unknown firrtl type"), -2;
    typeStr = typeStr.drop_front().drop_back();
    int32_t width;
    if (typeStr.getAsInteger(10, width) || width < 0)
      return parser.emitError(parser.getNameLoc(), "unknown width"), -2;
    return width;
  };

  if (typeStr.startswith("sint")) {
    typeStr = typeStr.drop_front(strlen("sint"));
    auto width = parseWidth();
    return SIntType::get(getContext(), width);
  }

  if (typeStr.startswith("uint")) {
    typeStr = typeStr.drop_front(strlen("uint"));
    auto width = parseWidth();
    return UIntType::get(getContext(), width);
  }

  if (typeStr.startswith("analog")) {
    typeStr = typeStr.drop_front(strlen("analog"));
    auto width = parseWidth();
    return width == -2 ? Type() : AnalogType::get(getContext(), width);
  }

  parser.emitError(parser.getNameLoc(), "unknown firrtl type");
  return Type();
}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  auto printWidthQualifier = [&](Optional<int32_t> width) {
    if (width)
      os.getStream() << '<' << width.getValue() << '>';
  };

  switch (type.getKind()) {
  default:
    assert(0 && "unknown dialect type to print");
  case FIRRTLTypes::Clock:
    os.getStream() << "clock";
    break;
  case FIRRTLTypes::Reset:
    os.getStream() << "reset";
    break;

  // Width Qualified Types.
  case FIRRTLTypes::SInt:
    os.getStream() << "sint";
    printWidthQualifier(type.cast<SIntType>().getWidth());
    break;
  case FIRRTLTypes::UInt:
    os.getStream() << "uint";
    printWidthQualifier(type.cast<UIntType>().getWidth());
    break;
  case FIRRTLTypes::Analog:
    os.getStream() << "analog";
    printWidthQualifier(type.cast<AnalogType>().getWidth());
    break;
  }
}

//===----------------------------------------------------------------------===//
// WidthTypeStorage
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
  return Base::get(context, FIRRTLTypes::SInt, width);
}

UIntType UIntType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, FIRRTLTypes::UInt, width);
}

/// Get an with a known width, or -1 for unknown.
AnalogType AnalogType::get(MLIRContext *context, int32_t width) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, FIRRTLTypes::Analog, width);
}