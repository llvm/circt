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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.indices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.memref(),

                                                SmallVector<Value>{finalIdx});
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.indices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.value(), adaptor.memref(), SmallVector<Value>{finalIdx});
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

struct ReturnOpConversion : public OpConversionPattern<mlir::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, adaptor.operands());
    return success();
  }
};

struct CondBranchOpConversion : public OpConversionPattern<mlir::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
        op, adaptor.condition(), adaptor.trueDestOperands(),
        adaptor.falseDestOperands(), op.trueDest(), op.falseDest());
    return success();
  }
};

struct FlattenMemRefPass : public FlattenMemRefBase<FlattenMemRefPass> {
public:
  void runOnOperation() override {

    auto *ctx = &getContext();
    TypeConverter typeConverter;
    // Add default conversion for all types generically.
    typeConverter.addConversion([](Type type) { return type; });
    // Add specific conversion for memref types.
    typeConverter.addConversion([](MemRefType memref) {
      if (isUniDimensional(memref))
        return memref;
      return MemRefType::get(
          llvm::SmallVector<int64_t>{memref.getNumElements()},
          memref.getElementType());
    });

    RewritePatternSet patterns(ctx);
    patterns.add<LoadOpConversion, StoreOpConversion, AllocOpConversion,
                 ReturnOpConversion, CondBranchOpConversion>(ctx);
    populateFuncOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addDynamicallyLegalOp<memref::AllocOp>(
        [](memref::AllocOp op) { return isUniDimensional(op.getType()); });
    target.addDynamicallyLegalOp<memref::StoreOp>(
        [](memref::StoreOp op) { return op.getIndices().size() == 1; });
    target.addDynamicallyLegalOp<memref::LoadOp>(
        [](memref::LoadOp op) { return op.getIndices().size() == 1; });
    target.addDynamicallyLegalOp<mlir::ReturnOp>(
        [](mlir::ReturnOp op) { return !hasMultiDimMemRef(op.operands()); });
    target.addDynamicallyLegalOp<mlir::CondBranchOp>([](mlir::CondBranchOp op) {
      return !hasMultiDimMemRef(op.trueDestOperands()) &&
             !hasMultiDimMemRef(op.falseDestOperands());
    });
    target.addDynamicallyLegalOp<mlir::FuncOp>([](mlir::FuncOp op) {
      auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
        return hasMultiDimMemRef(block.getArguments());
      });

      auto resultsConverted =
          llvm::all_of(op.getType().getResults(), [](Type type) {
            if (auto memref = type.dyn_cast<MemRefType>())
              return isUniDimensional(memref);
            return true;
          });

      return argsConverted && resultsConverted;
    });

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
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
