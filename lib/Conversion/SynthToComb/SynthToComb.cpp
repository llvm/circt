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
    // Only handle 1 or 3-input majority inverter for now.
    if (op.getNumOperands() > 3)
      return failure();

    auto getOperand = [&](unsigned idx) {
      auto input = op.getInputs()[idx];
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

    assert(op.getNumOperands() == 3 && "Expected 3 operands for majority op");
    SmallVector<Value, 3> inputs;
    for (size_t i = 0; i < 3; ++i)
      inputs.push_back(getOperand(i));

    // MAJ(x, y, z) = x & y | x & z | y & z
    auto getProduct = [&](unsigned idx1, unsigned idx2) {
      return rewriter.createOrFold<comb::AndOp>(
          op.getLoc(), ValueRange{inputs[idx1], inputs[idx2]}, true);
    };

    SmallVector<Value, 3> operands;
    operands.push_back(getProduct(0, 1));
    operands.push_back(getProduct(0, 2));
    operands.push_back(getProduct(1, 2));

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
  patterns.add<SynthAndInverterOpConversion, SynthMajorityInverterOpConversion>(
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
