//===- IbisConvertHandshakeToDCPass.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Conversion/HandshakeToDC.h"
#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Conversion/HandshakeToDC.h"

namespace circt {
namespace ibis {
#define GEN_PASS_DEF_IBISCONVERTHANDSHAKETODC
#include "circt/Dialect/Ibis/IbisPasses.h.inc"
} // namespace ibis
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace ibis;

namespace {

struct ConvertHandshakeToDCPass
    : public circt::ibis::impl::IbisConvertHandshakeToDCBase<
          ConvertHandshakeToDCPass> {
  void runOnOperation() override;
};

class ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;
  using OpAdaptor = typename ReturnOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ibis::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct StaticBlockOpConversion
    : public OpConversionPattern<IsolatedStaticBlockOp> {
  using OpConversionPattern<IsolatedStaticBlockOp>::OpConversionPattern;
  using OpAdaptor = typename IsolatedStaticBlockOp::Adaptor;

  LogicalResult
  matchAndRewrite(IsolatedStaticBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    auto dcBlock = rewriter.create<DCBlockOp>(op.getLoc(), resultTypes,
                                              adaptor.getOperands());
    rewriter.eraseOp(dcBlock.getBodyBlock()->getTerminator());
    rewriter.mergeBlocks(op.getBodyBlock(), dcBlock.getBodyBlock(),
                         dcBlock.getBodyBlock()->getArguments());

    rewriter.replaceOp(op, dcBlock.getResults());
    return success();
  }
};

} // anonymous namespace

static bool isDCType(Type t) { return isa<dc::ValueType, dc::TokenType>(t); }

static bool isDCTypedOp(Operation *op) {
  return llvm::all_of(op->getOperandTypes(), isDCType) &&
         llvm::all_of(op->getResultTypes(), isDCType);
}

void ConvertHandshakeToDCPass::runOnOperation() {
  ibis::ClassOp classOp = getOperation();
  auto targetModifier = [&](mlir::ConversionTarget &target) {
    target.addDynamicallyLegalOp<ibis::DataflowMethodOp>(
        [](ibis::DataflowMethodOp op) {
          auto methodLikeOp = cast<MethodLikeOpInterface>(op.getOperation());
          return llvm::all_of(methodLikeOp.getArgumentTypes(), isDCType) &&
                 llvm::all_of(methodLikeOp.getResultTypes(), isDCType);
        });
    target.addDynamicallyLegalOp<ibis::ReturnOp>(isDCTypedOp);
    target.addLegalDialect<hw::HWDialect, ibis::IbisDialect>();
    target.addIllegalOp<ibis::IsolatedStaticBlockOp>();

    // ibis.sblock.dc ops are recursively legal - we're only considering the
    // DataflowMethodOp's region for conversion.
    target.addLegalOp<ibis::DCBlockOp>();
    target.markOpRecursivelyLegal<ibis::DCBlockOp>();
  };

  auto patternBuilder = [&](TypeConverter &typeConverter,
                            handshaketodc::ConvertedOps &convertedOps,
                            RewritePatternSet &patterns) {
    patterns
        .add<TypeOpConversionPattern<DataflowMethodOp>,
             TypeOpConversionPattern<ReturnOp>,
             TypeOpConversionPattern<StaticBlockOp>, StaticBlockOpConversion>(
            typeConverter, classOp.getContext());
  };

  LogicalResult res =
      handshaketodc::runHandshakeToDC(classOp, patternBuilder, targetModifier);
  if (failed(res))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createConvertHandshakeToDCPass() {
  return std::make_unique<ConvertHandshakeToDCPass>();
}
