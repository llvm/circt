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

struct SynthMajorityInverterOpConversion
    : OpConversionPattern<synth::mig::MajorityInverterOp> {
  using OpConversionPattern<
      synth::mig::MajorityInverterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(synth::mig::MajorityInverterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto getOperand = [&](unsigned idx) {
      auto input = adaptor.getInputs()[idx];
      if (!op.getInverted()[idx])
        return input;
      auto width = input.getType().getIntOrFloatBitWidth();
      auto allOnes = hw::ConstantOp::create(rewriter, op.getLoc(),
                                            APInt::getAllOnes(width));
      return rewriter.createOrFold<comb::XorOp>(op.getLoc(), input, allOnes,
                                                true);
    };

    if (op.getNumOperands() == 1) {
      rewriter.replaceOp(op, getOperand(0));
      return success();
    }

    SmallVector<Value> inputs;
    inputs.reserve(op.getNumOperands());
    for (size_t i = 0, e = op.getNumOperands(); i < e; ++i)
      inputs.push_back(getOperand(i));

    // MAJ_n(x_0, ..., x_n) is the OR of all conjunctions over threshold-sized
    // subsets, where threshold = floor(n / 2) + 1.
    auto getProduct = [&](ArrayRef<unsigned> indices) {
      SmallVector<Value> productOperands;
      productOperands.reserve(indices.size());
      for (auto idx : indices)
        productOperands.push_back(inputs[idx]);
      return rewriter.createOrFold<comb::AndOp>(op.getLoc(), productOperands,
                                                true);
    };

    SmallVector<Value> operands;
    SmallVector<unsigned> subset;
    const unsigned threshold = op.getNumOperands() / 2 + 1;

    auto enumerateProducts = [&](auto &&self, unsigned start) -> void {
      if (subset.size() == threshold) {
        operands.push_back(getProduct(subset));
        return;
      }

      const unsigned remaining = threshold - subset.size();
      assert(start + remaining <= op.getNumOperands() &&
             "Not enough operands left to reach threshold");
      for (unsigned i = start, e = op.getNumOperands() - remaining; i <= e;
           ++i) {
        subset.push_back(i);
        self(self, i + 1);
        subset.pop_back();
      }
    };
    enumerateProducts(enumerateProducts, 0);

    rewriter.replaceOp(
        op, rewriter.createOrFold<comb::OrOp>(op.getLoc(), operands, true));
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
  patterns.add<SynthChoiceOpConversion, SynthAndInverterOpConversion,
               SynthMajorityInverterOpConversion>(patterns.getContext());
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
