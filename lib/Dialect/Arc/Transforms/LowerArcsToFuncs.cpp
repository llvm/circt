//===- LowerArcsToFuncs.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-arcs-to-funcs"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERARCSTOFUNCS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerArcsToFuncsPass
    : public arc::impl::LowerArcsToFuncsBase<LowerArcsToFuncsPass> {

  LogicalResult lowerToFuncs();
  void runOnOperation() override;
};

struct DefineOpLowering : public OpConversionPattern<arc::DefineOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::DefineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                           op.getFunctionType());
    func->setAttr(
        "llvm.linkage",
        LLVM::LinkageAttr::get(getContext(), LLVM::linkage::Linkage::Internal));
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct OutputOpLowering : public OpConversionPattern<arc::OutputOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOutputs());
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<arc::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, newResultTypes, op.getArcAttr(), adaptor.getInputs());
    return success();
  }
};

struct StateOpLowering : public OpConversionPattern<arc::StateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arc::StateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, newResultTypes, op.getArcAttr(), adaptor.getInputs());
    return success();
  }
};

} // namespace

static void populateLegality(ConversionTarget &target) {
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  target.addIllegalOp<arc::CallOp>();
  target.addIllegalOp<arc::DefineOp>();
  target.addIllegalOp<arc::OutputOp>();
  target.addIllegalOp<arc::StateOp>();
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  auto *context = patterns.getContext();
  patterns
      .add<CallOpLowering, DefineOpLowering, OutputOpLowering, StateOpLowering>(
          typeConverter, context);
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](Type type) { return type; });
}

LogicalResult LowerArcsToFuncsPass::lowerToFuncs() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering arcs to funcs\n");
  ConversionTarget target(getContext());
  TypeConverter converter;
  RewritePatternSet patterns(&getContext());
  populateLegality(target);
  populateTypeConversion(converter);
  populateOpConversion(patterns, converter);
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

void LowerArcsToFuncsPass::runOnOperation() {
  if (failed(lowerToFuncs()))
    return signalPassFailure();
}

std::unique_ptr<Pass> arc::createLowerArcsToFuncsPass() {
  return std::make_unique<LowerArcsToFuncsPass>();
}
