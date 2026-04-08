//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Synth to Comb Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SynthToComb.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTSYNTHTOCOMB
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct SynthChoiceOpConversion : OpConversionPattern<synth::ChoiceOp> {
  using OpConversionPattern<synth::ChoiceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(synth::ChoiceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use the first input as the output, and ignore the rest.
    rewriter.replaceOp(op, adaptor.getInputs().front());
    return success();
  }
};

struct SynthAndInverterOpConversion
    : OpConversionPattern<synth::aig::AndInverterOp> {
  using OpConversionPattern<synth::aig::AndInverterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(synth::aig::AndInverterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert to comb.and + comb.xor + hw.constant
    auto width = op.getResult().getType().getIntOrFloatBitWidth();
    auto allOnes =
        hw::ConstantOp::create(rewriter, op.getLoc(), APInt::getAllOnes(width));
    SmallVector<Value> operands;
    operands.reserve(op.getNumOperands());
    for (auto [input, inverted] : llvm::zip(op.getOperands(), op.getInverted()))
      operands.push_back(inverted ? rewriter.createOrFold<comb::XorOp>(
                                        op.getLoc(), input, allOnes, true)
                                  : input);
    // NOTE: Use createOrFold to avoid creating a new operation if possible.
    rewriter.replaceOp(
        op, rewriter.createOrFold<comb::AndOp>(op.getLoc(), operands, true));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Synth to Comb pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertSynthToCombPass
    : public impl::ConvertSynthToCombBase<ConvertSynthToCombPass> {

  void runOnOperation() override;
  using ConvertSynthToCombBase<ConvertSynthToCombPass>::ConvertSynthToCombBase;
};
} // namespace

static void populateSynthToCombConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<SynthChoiceOpConversion, SynthAndInverterOpConversion>(
      patterns.getContext());
}

void ConvertSynthToCombPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
  target.addIllegalDialect<synth::SynthDialect>();

  RewritePatternSet patterns(&getContext());
  populateSynthToCombConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
