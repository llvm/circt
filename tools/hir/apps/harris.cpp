#include "../HIRGen.h"
using namespace mlir;
using namespace hir;
void calcOutputDims(int outputDims[2], int inputDims[2], int kernelDims[2]) {
  // FIXME : Calculate outputDims based on inputDims and kernelDims.
  outputDims[0] = inputDims[0];
  outputDims[1] = inputDims[1];
};

void emitHarrisHelpers(
    mlir::OpBuilder &builder, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module, int imgDims[2], int kernelDims[2],
    mlir::SmallVectorImpl<mlir::SmallVector<float, 2>> &kernelX,
    mlir::SmallVectorImpl<mlir::SmallVector<float, 2>> &kernelY) {

  int iDims[2];
  calcOutputDims(iDims, imgDims, kernelDims);

  emitLineBuffer(builder, context, module, "line_buffer1", imgDims, kernelDims);

  emitDotProduct2D(builder, context, module, kernelX);

  emitDotProduct2D(builder, context, module, kernelY);
  emitMultiply(builder, context, module);
  emitMultiply(builder, context, module);
  emitMultiply(builder, context, module);
  emitLineBuffer(builder, context, module, "line_buffer2", iDims, kernelDims);
  emitLineBuffer(builder, context, module, "line_buffer2", iDims, kernelDims);
  emitLineBuffer(builder, context, module, "line_buffer2", iDims, kernelDims);
  emitSum2D(builder, context, module, kernelDims);
  emitSum2D(builder, context, module, kernelDims);
  emitSum2D(builder, context, module, kernelDims);
  emitDet(builder, context, module);
  emitTrace(builder, context, module);
  emitHarris(builder, context, module);
}

void emitHarris(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                mlir::OwningModuleRef &module) {
  SmallVector<SmallVector<float, 2>, 2> kernelX = {
      {1 / 12, 0, -1 / 12}, {1 / 12, 0, -1 / 12}, {1 / 12, 0, -1 / 12}};
  SmallVector<SmallVector<float, 2>, 2> kernelY = {
      {1 / 12, 1 / 12, 1 / 12}, {0, 0, 0}, {-1 / 12, -1 / 12, -1 / 12}};
  int imgDims[2] = {16, 16};
  int kernelDims[2] = {2, 2};
  emitHarrisHelpers(builder, context, module, imgDims, kernelDims, kernelX,
                    kernelY);
}
