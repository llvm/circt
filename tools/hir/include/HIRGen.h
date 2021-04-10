#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"
class ExprTree;
int emitMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module);

void emitLineBuffer(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                    mlir::OwningModuleRef &module, mlir::StringRef funcName,
                    int imgDims[2], int kernelDims[2]);

void emitDotProduct2D(
    mlir::OpBuilder &builder, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module,
    mlir::SmallVectorImpl<mlir::SmallVector<float, 2>> &kernel);

void emitMultiply(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                  mlir::OwningModuleRef &module);
void emitSum2D(mlir::OpBuilder &builder, mlir::MLIRContext &context,
               mlir::OwningModuleRef &module, int dims[2]);
void emitDet(mlir::OpBuilder &builder, mlir::MLIRContext &context,
             mlir::OwningModuleRef &module);
void emitTrace(mlir::OpBuilder &builder, mlir::MLIRContext &context,
               mlir::OwningModuleRef &module);
void emitHarris(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                mlir::OwningModuleRef &module);

void emitExpr(mlir::OpBuilder &builder, mlir::MLIRContext &context,
              mlir::OwningModuleRef &module, ExprTree expr);
