//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to Datapath Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToDatapath.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTODATAPATH
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
// add(a1, a2, ...) -> add(compress(a1, a2, ...))
struct CombAddOpConversion : OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto width = op.getType().getIntOrFloatBitWidth();
    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(), 0);
      return success();
    }

    // Reduce to two values (carry,save)
    auto results = datapath::CompressOp::create(rewriter, op.getLoc(),
                                                op.getOperands(), 2);
    // carry+saved
    rewriter.replaceOpWithNewOp<AddOp>(op, results.getResults(), true);
    return success();
  }
};

// mul(a,b) -> add(pp(a,b))
// multi-input adder will be converted to a compressor by other pattern
struct CombMulOpConversion : OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: support for variadic multipliers
    if (adaptor.getInputs().size() != 2)
      return failure();

    auto width = op.getType().getIntOrFloatBitWidth();
    // Create partial product rows - number of rows == width
    auto pp = datapath::PartialProductOp::create(rewriter, op.getLoc(),
                                                 op.getInputs(), width);
    // Sum partial products
    rewriter.replaceOpWithNewOp<AddOp>(op, pp.getResults(), true);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Datapath pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToDatapathPass
    : public impl::ConvertCombToDatapathBase<ConvertCombToDatapathPass> {
  void runOnOperation() override;
  using ConvertCombToDatapathBase<
      ConvertCombToDatapathPass>::ConvertCombToDatapathBase;
};
} // namespace

static void
populateCombToDatapathConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<CombAddOpConversion, CombMulOpConversion>(patterns.getContext());
}

void ConvertCombToDatapathPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<datapath::DatapathDialect, comb::CombDialect,
                         hw::HWDialect>();

  // Permit 2-input adders (carry-propagate adders)
  target.addDynamicallyLegalOp<comb::AddOp>(
      [](comb::AddOp op) { return op.getNumOperands() <= 2; });
  // TODO: determine lowering of multi-input multipliers
  target.addDynamicallyLegalOp<comb::MulOp>(
      [](comb::MulOp op) { return op.getNumOperands() > 2; });

  RewritePatternSet patterns(&getContext());
  populateCombToDatapathConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
