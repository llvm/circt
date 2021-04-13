#include "../include/HIRGen.h"

void emitMultiply(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                  mlir::OwningModuleRef &module);

void emitDotProduct2D(
    mlir::OpBuilder &builder, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module,
    mlir::SmallVectorImpl<mlir::SmallVector<float, 2>> &kernel);
