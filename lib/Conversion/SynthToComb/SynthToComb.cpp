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

// Return a value that represents the inverted input if `inverted` is true,
static Value materializeInvertedInput(Location loc, Value input, bool inverted,
                                      ConversionPatternRewriter &rewriter,
                                      hw::ConstantOp &allOnes) {
  if (!inverted)
    return input;
  auto width = input.getType().getIntOrFloatBitWidth();
  if (!allOnes)
    allOnes = hw::ConstantOp::create(rewriter, loc, APInt::getAllOnes(width));
  return rewriter.createOrFold<comb::XorOp>(loc, input, allOnes, true);
}

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

template <typename SynthOp>
struct SynthInverterOpConversion : OpConversionPattern<SynthOp> {
  using OpConversionPattern<SynthOp>::OpConversionPattern;
  // Subclasses provide the target comb op after generic input inversion has
  // been materialized.
  virtual Value createOp(Location loc, ArrayRef<Value> inputs,
                         ConversionPatternRewriter &rewriter) const = 0;

  virtual LogicalResult
  matchAndRewrite(SynthOp op, typename SynthOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    operands.reserve(op.getNumOperands());
    hw::ConstantOp allOnes;
    for (auto [input, inverted] :
         llvm::zip(adaptor.getOperands(), op.getInverted()))
      operands.push_back(materializeInvertedInput(op.getLoc(), input, inverted,
                                                  rewriter, allOnes));
    // `createOp` now only needs to encode the core boolean operator.
    rewriter.replaceOp(op, createOp(op.getLoc(), operands, rewriter));
    return success();
  }
};

struct SynthAndInverterOpConversion
    : SynthInverterOpConversion<synth::aig::AndInverterOp> {
  using SynthInverterOpConversion<
      synth::aig::AndInverterOp>::SynthInverterOpConversion;
  Value createOp(Location loc, ArrayRef<Value> inputs,
                 ConversionPatternRewriter &rewriter) const override {
    return rewriter.createOrFold<comb::AndOp>(loc, inputs, true);
  }
};

struct SynthXorInverterOpConversion
    : SynthInverterOpConversion<synth::XorInverterOp> {
  using SynthInverterOpConversion<
      synth::XorInverterOp>::SynthInverterOpConversion;
  Value createOp(Location loc, ArrayRef<Value> inputs,
                 ConversionPatternRewriter &rewriter) const override {
    return rewriter.createOrFold<comb::XorOp>(loc, inputs, true);
  }
};

struct SynthDotOpConversion : SynthInverterOpConversion<synth::DotOp> {
  using SynthInverterOpConversion<synth::DotOp>::SynthInverterOpConversion;
  Value createOp(Location loc, ArrayRef<Value> inputs,
                 ConversionPatternRewriter &rewriter) const override {
    assert(inputs.size() == 3 && "expected exactly three inputs");
    auto xy =
        rewriter.createOrFold<comb::AndOp>(loc, inputs[0], inputs[1], true);
    auto zOrXy = rewriter.createOrFold<comb::OrOp>(loc, inputs[2], xy, true);
    return rewriter.createOrFold<comb::XorOp>(loc, inputs[0], zOrXy, true);
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
  patterns.add<SynthChoiceOpConversion, SynthAndInverterOpConversion,
               SynthXorInverterOpConversion, SynthDotOpConversion>(
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
