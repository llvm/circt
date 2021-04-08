#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
int emitMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module);

void emitLineBuffer(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                    mlir::OwningModuleRef &module, mlir::StringRef funcName);
