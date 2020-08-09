#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hir {
HIRDialect::HIRDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<TimeType, MemoryInterfaceType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
      >();
}

// Types
Type HIRDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword)) {
    return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
  }

  if (typeKeyword == TimeType::getKeyword()) {
    return TimeType::get(getContext());
  }
  if (typeKeyword == MemoryInterfaceType::getKeyword()) {
    return MemoryInterfaceType::get(getContext());
  }
  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

void HIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (TimeType time = type.dyn_cast<TimeType>()) {
    printer << time.getKeyword();
  }
  if (MemoryInterfaceType mem_interface = type.dyn_cast<MemoryInterfaceType>()) {
    printer << mem_interface.getKeyword();
  }
}



} // namespace hir
} // namespace mlir
