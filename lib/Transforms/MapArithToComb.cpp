//===- MapArithToComb.cpp - Arith-to-comb mapping pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MapArithToComb pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;

template <typename TFrom, typename TTo>
class OneToOnePattern : public OpConversionPattern<TFrom> {
public:
  OneToOnePattern(MLIRContext *context) : OpConversionPattern<TFrom>(context) {}
  using OpAdaptor = typename TFrom::Adaptor;

  LogicalResult
  matchAndRewrite(TFrom op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TTo>(op, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

class ExtSConvertionPattern : public OpConversionPattern<arith::ExtSIOp> {
public:
  ExtSConvertionPattern(MLIRContext *context)
      : OpConversionPattern<arith::ExtSIOp>(context) {}
  using OpAdaptor = typename arith::ExtSIOp::Adaptor;

  LogicalResult
  matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    size_t outWidth = op.getType().getIntOrFloatBitWidth();
    rewriter.replaceOp(op, comb::createOrFoldSExt(
                               op.getLoc(), op.getOperand(),
                               rewriter.getIntegerType(outWidth), rewriter));
    return success();
  }
};

class ExtZConvertionPattern : public OpConversionPattern<arith::ExtUIOp> {
public:
  ExtZConvertionPattern(MLIRContext *context)
      : OpConversionPattern<arith::ExtUIOp>(context) {}
  using OpAdaptor = typename arith::ExtUIOp::Adaptor;

  LogicalResult
  matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    size_t outWidth = op.getOut().getType().getIntOrFloatBitWidth();
    size_t inWidth = adaptor.getIn().getType().getIntOrFloatBitWidth();

    llvm::dbgs() << "out: " << outWidth << " in: " << inWidth << "\n";

    rewriter.replaceOp(
        op,
        {rewriter.create<comb::ConcatOp>(
            loc,
            rewriter.create<hw::ConstantOp>(loc, APInt(outWidth - inWidth, 0)),
            adaptor.getIn())});
    return success();
  }
};

class TruncateConversionPattern : public OpConversionPattern<arith::TruncIOp> {
public:
  TruncateConversionPattern(MLIRContext *context)
      : OpConversionPattern<arith::TruncIOp>(context) {}
  using OpAdaptor = typename arith::TruncIOp::Adaptor;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    size_t outWidth = op.getType().getIntOrFloatBitWidth();
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, adaptor.getIn(), 0,
                                                 outWidth);
    return success();
  }
};

namespace {
struct MapArithToCombPass : public MapArithToCombPassBase<MapArithToCombPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
    target.addIllegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(ctx);

    patterns.insert<OneToOnePattern<arith::AddIOp, comb::AddOp>,
                    OneToOnePattern<arith::SubIOp, comb::SubOp>,
                    OneToOnePattern<arith::MulIOp, comb::MulOp>,
                    OneToOnePattern<arith::DivSIOp, comb::DivSOp>,
                    OneToOnePattern<arith::DivUIOp, comb::DivUOp>,
                    OneToOnePattern<arith::RemSIOp, comb::ModSOp>,
                    OneToOnePattern<arith::RemUIOp, comb::ModUOp>,
                    OneToOnePattern<arith::AndIOp, comb::AndOp>,
                    OneToOnePattern<arith::OrIOp, comb::OrOp>,
                    OneToOnePattern<arith::XOrIOp, comb::XorOp>,
                    OneToOnePattern<arith::ShLIOp, comb::ShlOp>,
                    OneToOnePattern<arith::ShRSIOp, comb::ShrSOp>,
                    OneToOnePattern<arith::ShRUIOp, comb::ShrUOp>,
                    OneToOnePattern<arith::CmpIOp, comb::ICmpOp>,
                    OneToOnePattern<arith::SelectOp, comb::MuxOp>,
                    ExtSConvertionPattern, ExtZConvertionPattern,
                    TruncateConversionPattern>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::createMapArithToCombPass() {
  return std::make_unique<MapArithToCombPass>();
}
