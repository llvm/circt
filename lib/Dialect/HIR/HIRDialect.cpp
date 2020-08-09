#include "circt/Dialect/HIR/HIRDialect.h"
#include "circt/Dialect/HIR/HIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir{
  namespace hir{
    HIRDialect::HIRDialect(mlir::MLIRContext *context)
      : Dialect(getDialectNamespace(), context) {
        addTypes<TimeType>();
        addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
          >();
      }

    Type HIRDialect::parseType(DialectAsmParser &parser) const {
      llvm::StringRef typeKeyword;
      if (parser.parseKeyword(&typeKeyword)){
        return parser.emitError(parser.getNameLoc(), "unknown hir type"),Type();
      }

      if (typeKeyword == TimeType::getKeyword()){
        return TimeType::get(getContext());
      }

      return parser.emitError(parser.getNameLoc(), "unknown hir type"),Type();
    }

    void HIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
      if (TimeType time = type.dyn_cast<TimeType>()) {
        printer << time.getKeyword();
      }
    }
  }
}
