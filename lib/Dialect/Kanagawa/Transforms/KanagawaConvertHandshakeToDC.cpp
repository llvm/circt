//===- KanagawaConvertHandshakeToDCPass.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Dialect/HW/ConversionPatterns.h"
#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Conversion/HandshakeToDC.h"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWACONVERTHANDSHAKETODC
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace kanagawa;

namespace {

struct ConvertHandshakeToDCPass
    : public circt::kanagawa::impl::KanagawaConvertHandshakeToDCBase<
          ConvertHandshakeToDCPass> {
  void runOnOperation() override;
};

class ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;
  using OpAdaptor = typename ReturnOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<kanagawa::ReturnOp>(op, adaptor.getOperands());
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
    auto dcBlock = DCBlockOp::create(rewriter, op.getLoc(), resultTypes,
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
  kanagawa::ClassOp classOp = getOperation();
  auto targetModifier = [&](mlir::ConversionTarget &target) {
    target.addDynamicallyLegalOp<kanagawa::DataflowMethodOp>(
        [](kanagawa::DataflowMethodOp op) {
          auto methodLikeOp = cast<MethodLikeOpInterface>(op.getOperation());
          return llvm::all_of(methodLikeOp.getArgumentTypes(), isDCType) &&
                 llvm::all_of(methodLikeOp.getResultTypes(), isDCType);
        });
    target.addDynamicallyLegalOp<kanagawa::ReturnOp>(isDCTypedOp);
    target.addLegalDialect<hw::HWDialect, kanagawa::KanagawaDialect>();
    target.addIllegalOp<kanagawa::IsolatedStaticBlockOp>();

    // kanagawa.sblock.dc ops are recursively legal - we're only considering the
    // DataflowMethodOp's region for conversion.
    target.addLegalOp<kanagawa::DCBlockOp>();
    target.markOpRecursivelyLegal<kanagawa::DCBlockOp>();
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

  LogicalResult res = handshaketodc::runHandshakeToDC(
      classOp, {"clk", "rst"}, patternBuilder, targetModifier);
  if (failed(res))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::kanagawa::createConvertHandshakeToDCPass() {
  return std::make_unique<ConvertHandshakeToDCPass>();
}
