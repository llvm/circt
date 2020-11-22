//===- Types.cpp - RTL types code defs ------------------------------------===//
//
// Definitions for RTL data types. Anything which doesn't have to be public
// should go in here.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt::rtl;

Type ArrayType::parse(MLIRContext *ctxt, DialectAsmParser &p) {
  SmallVector<int64_t, 2> dims;
  Type inner;
  if (p.parseLess() || p.parseDimensionList(dims, /* allowDynamic */ false) ||
      p.parseType(inner) || p.parseGreater())
    return Type();
  if (dims.size() != 1) {
    p.emitError(p.getNameLoc(), "rtl.array only supports one dimension");
    return Type();
  }
  return get(ctxt, inner, dims[0]);
}

void ArrayType::print(DialectAsmPrinter &p) const {
  p << "array<" << getSize() << "x";
  p.printType(getInnerType());
  p << ">";
}

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTL/RTLTypes.cpp.inc"
