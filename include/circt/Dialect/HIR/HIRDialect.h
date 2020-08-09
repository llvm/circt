#ifndef HIR_HIRDIALECT_H
#define HIR_HIRDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
  namespace hir {
    class HIRDialect : public Dialect {
      public:
        explicit HIRDialect(MLIRContext *context);
        static ::llvm::StringRef getDialectNamespace() { return "hir"; }
        Type parseType(DialectAsmParser &parser) const override;
        void printType(Type type, DialectAsmPrinter &printer) const override;
    };
  } // namespace hir
} // namespace mlir

#endif // HIR_HIRDIALECT_H
