#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace hir;

HIRDialect::HIRDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<TimeType, IntType, MemoryInterfaceType, WireType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
      >();
}

// Types
Type HIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword))
    return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();

  if (typeKeyword == TimeType::getKeyword())
    return TimeType::get(getContext());

  if (typeKeyword == MemoryInterfaceType::getKeyword())
    return MemoryInterfaceType::get(getContext());

  if (typeKeyword == WireType::getKeyword())
    return WireType::get(getContext());

  if (typeKeyword == IntType::getKeyword())
    return IntType::get(getContext());

  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

void HIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (TimeType hirTime = type.dyn_cast<TimeType>()) {
    printer << hirTime.getKeyword();
    return;
  }
  if (MemoryInterfaceType mem_interface =
          type.dyn_cast<MemoryInterfaceType>()) {
    printer << mem_interface.getKeyword();
    return;
  }
  if (WireType wire = type.dyn_cast<WireType>()) {
    printer << wire.getKeyword();
    return;
  }
  if (IntType Int = type.dyn_cast<IntType>()) {
    printer << Int.getKeyword();
    return;
  }
}
