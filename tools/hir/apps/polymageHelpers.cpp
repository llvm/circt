#include "../include/HIRGen.h"
#include "circt/Dialect/HIR/helper.h"

using namespace mlir;
using namespace hir;

ExprTree *buildMultExpr(mlir::OpBuilder &builder, mlir::MLIRContext &context) {
  Type f32Ty = FloatType::getF32(&context);

  ExprTree *left = new ExprTree(0);
  ExprTree *right = new ExprTree(1);
  ExprTree::BinOp *binOp =
      new ExprTree::BinOp("f32Multiplier", left, right, 0, 4, f32Ty);
  ExprTree *expr = new ExprTree(binOp);
  return expr;
}

ExprTree *buildAddExpr(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                       ExprTree *left, ExprTree *right, int offset) {
  Type f32Ty = FloatType::getF32(&context);

  ExprTree::BinOp *binOp =
      new ExprTree::BinOp("f32Adder", left, right, offset, 4, f32Ty);
  ExprTree *expr = new ExprTree(binOp);
  return expr;
}

void emitMultiply(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                  mlir::OwningModuleRef &module) {

  Type f32Ty = FloatType::getF32(&context);
  ExprTree *expr = buildMultExpr(builder, context);
  SmallVector<Type, 2> argTypes = {f32Ty, f32Ty};
  emitExpr(builder, context, module, "multiply", argTypes, expr);
}

ExprTree *buildReductionTree(mlir::OpBuilder &builder,
                             mlir::MLIRContext &context, int n, ExprTree **arr,
                             int offset) {
  if (n == 1)
    return arr[0];

  if (n % 2 == 0) {
    ExprTree *arrNext[n / 2];
    for (int i = 0; i < n; i += 2) {
      ExprTree *left = arr[i];
      ExprTree *right = arr[i + 1];
      arrNext[i / 2] = buildAddExpr(builder, context, left, right, offset);
    }
    return buildReductionTree(builder, context, n / 2, arrNext, offset + 4);
  }
  ExprTree *last = arr[n - 1];
  n = n - 1;
  ExprTree *arrNext[n / 2 + 1];
  for (int i = 0; i < n; i += 2) {
    ExprTree *left = arr[i];
    ExprTree *right = arr[i + 1];
    arrNext[i / 2] = buildAddExpr(builder, context, left, right, offset);
  }
  ExprTree* lastDelayed = new ExprTree(new ExprTree::UnaryOp("hir.delay",last,
  arrNext[n / 2] = last; // don't worry about pipeline balancing here.
  return buildReductionTree(builder, context, n / 2 + 1, arrNext);
}

ExprTree *buildDotProduct2DExpr(mlir::OpBuilder &builder,
                                mlir::MLIRContext &context, int nRows,
                                int nCols) {

  Type f32Ty = FloatType::getF32(&context);
  ExprTree *input = new ExprTree(0);

  ExprTree *selectOpExprs[nRows * nCols];
  for (int i = 0; i < nRows; i++) {
    for (int j = 0; j < nCols; j++) {
      selectOpExprs[i * nCols + j] =
          new ExprTree(new ExprTree::SelectOp(input, {i, j}, f32Ty));
    }
  }

  return buildReductionTree(builder, context, nRows * nCols, selectOpExprs);
}

void emitDotProduct2D(
    mlir::OpBuilder &builder, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module,
    mlir::SmallVectorImpl<mlir::SmallVector<float, 2>> &kernel) {

  int nRows = kernel.size();
  int nCols = kernel[0].size();

  Type f32Ty = FloatType::getF32(&context);
  ArrayType arrayTy = ArrayType::get(&context, {nRows, nCols}, f32Ty);
  Type timeTy = helper::getTimeType(&context);
  GroupType groupTy =
      GroupType::get(&context, {timeTy, arrayTy}, {Attribute()});
  ExprTree *expr = buildDotProduct2DExpr(builder, context, nRows, nCols);
  SmallVector<Type, 2> argTypes = {groupTy};
  emitExpr(builder, context, module, "dotProduct2D", argTypes, expr);
}
