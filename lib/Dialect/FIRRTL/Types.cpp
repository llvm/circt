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
  StringRef tyData = parser.getFullSymbolSpec();

  if (tyData == "sint")
    return SIntType::get(getContext());
  if (tyData == "uint")
    return UIntType::get(getContext());
  if (tyData == "clock")
    return ClockType::get(getContext());
  if (tyData == "reset")
    return ResetType::get(getContext());
  if (tyData.startswith("analog")) {
    tyData = tyData.drop_front(strlen("analog"));
    if (tyData.empty())
      return AnalogType::get(getContext());
    if (tyData.front() != '<' || tyData.back() != '>') {
      parser.emitError(parser.getNameLoc(), "unknown firrtl type");
      return Type();
    }
    tyData = tyData.drop_front().drop_back();
    int32_t width;
    if (!tyData.getAsInteger(10, width) && width >= 0)
      return AnalogType::get(width, getContext());
    parser.emitError(parser.getNameLoc(), "unknown width");
    return Type();
  }

  parser.emitError(parser.getNameLoc(), "unknown firrtl type");
  return Type();
}

void FIRRTLDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  default:
    assert(0 && "unknown dialect type to print");
  case FIRRTLTypes::SInt:
    os.getStream() << "sint";
    break;
  case FIRRTLTypes::UInt:
    os.getStream() << "uint";
    break;
  case FIRRTLTypes::Clock:
    os.getStream() << "clock";
    break;
  case FIRRTLTypes::Reset:
    os.getStream() << "reset";
    break;

  // Derived types.
  case FIRRTLTypes::Analog:
    os.getStream() << "analog";
    if (auto width = type.cast<AnalogType>().getWidth())
      os.getStream() << '<' << width.getValue() << '>';
    break;
  }
}

//===----------------------------------------------------------------------===//
// AnalogType
//===----------------------------------------------------------------------===//

namespace spt {
namespace firrtl {
namespace detail {
struct AnalogTypeStorage : mlir::TypeStorage {
  AnalogTypeStorage(int32_t width) : width(width) {}
  using KeyTy = int32_t;

  bool operator==(const KeyTy &key) const { return key == width; }

  static AnalogTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<AnalogTypeStorage>()) AnalogTypeStorage(key);
  }

  int32_t width;
};
} // namespace detail
} // namespace firrtl
} // namespace spt

/// Get an AnalogType with a known width, or -1 for unknown.
AnalogType AnalogType::get(int32_t width, MLIRContext *context) {
  assert(width >= -1 && "unknown width");
  return Base::get(context, FIRRTLTypes::Analog, width);
}

Optional<int32_t> AnalogType::getWidth() const {
  int width = getImpl()->width;
  if (width < 0)
    return None;
  return width;
}
