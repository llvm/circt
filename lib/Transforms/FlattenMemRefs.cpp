//===- FlattenMemRefs.cpp - MemRef flattening pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MemRef flattening pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;

static bool isUniDimensional(MemRefType memref) {
  return memref.getShape().size() == 1;
}

// Flatten indices by generating the product of the i'th index and the [0:i-1]
// shapes, for each index, and then summing these.
static Value flattenIndices(ConversionPatternRewriter &rewriter, Operation *op,
                            ValueRange indices, MemRefType memrefType) {
  Location loc = op->getLoc();
  Value finalIdx = indices.front();
  for (auto memIdx : llvm::enumerate(indices.drop_front())) {
    Value partialIdx = memIdx.value();
    for (unsigned i = 0; i <= memIdx.index(); ++i) {
      int64_t dimSize = memrefType.getShape()[i];
      if (llvm::isPowerOf2_64(dimSize)) {
        auto constant =
            rewriter
                .create<arith::ConstantOp>(
                    loc, rewriter.getIndexAttr(llvm::Log2_64(dimSize)))
                .getResult();
        partialIdx = rewriter.create<arith::ShLIOp>(loc, partialIdx, constant)
                         .getResult();
      } else {
        auto constant =
            rewriter
                .create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dimSize))
                .getResult();
        partialIdx = rewriter.create<arith::MulIOp>(loc, partialIdx, constant)
                         .getResult();
      }
    }
    auto sumOp = rewriter.create<arith::AddIOp>(loc, finalIdx, partialIdx);
    finalIdx = sumOp.getResult();
  }
  return finalIdx;
}

static bool hasMultiDimMemRef(ValueRange values) {
  return llvm::any_of(values, [](Value v) {
    auto memref = v.getType().dyn_cast<MemRefType>();
    if (!memref)
      return false;
    return !isUniDimensional(memref);
  });
}

namespace {

struct LoadOpConversion : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, op.getIndices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.memref(),
                                                SmallVector<Value>{finalIdx});
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, op.getIndices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.memref(), SmallVector<Value>{finalIdx});
    return success();
  }
};

struct AllocOpConversion : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (isUniDimensional(type) || !type.hasStaticShape())
      return failure();
    MemRefType newType = MemRefType::get(
        SmallVector<int64_t>{type.getNumElements()}, type.getElementType());
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newType);
    return success();
  }
};

struct MemRefTypeConversion : public OpConversionPattern<mlir::FuncOp> {
  using OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {

    if (llvm::none_of(op.getBlocks(), [](auto &block) {
          return hasMultiDimMemRef(block.getArguments());
        }))
      return failure();

    bool err = false;
    rewriter.updateRootInPlace(op, [&] {
      if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter)))
        err = true;
    });
    return err ? failure() : success();
  }
};

struct FlattenMemRefPass : public FlattenMemRefBase<FlattenMemRefPass> {
public:
  void runOnOperation() override {

    auto ctx = &getContext();
    TypeConverter typeConverter;
    typeConverter.addConversion([&](MemRefType memref) {
      if (isUniDimensional(memref))
        return memref;
      return MemRefType::get(
          llvm::SmallVector<int64_t>{memref.getNumElements()},
          memref.getElementType());
    });

    RewritePatternSet patterns(ctx);
    patterns.add<LoadOpConversion, StoreOpConversion, AllocOpConversion>(ctx);
    patterns.add<MemRefTypeConversion>(typeConverter, ctx);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
            .failed()) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createFlattenMemRefPass() {
  return std::make_unique<FlattenMemRefPass>();
}
} // namespace circt
