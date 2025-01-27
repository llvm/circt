//===- AIGToComb.cpp - AIG to Comb Conversion Pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main AIG to Comb Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AIGToComb.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTAIGTOCOMB
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct AIGAndInverterOpConversion : OpConversionPattern<aig::AndInverterOp> {
  using OpConversionPattern<aig::AndInverterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(aig::AndInverterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert to comb.and + comb.xor + hw.constant
    auto width = op.getResult().getType().getIntOrFloatBitWidth();
    auto allOnes =
        rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getAllOnes(width));
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
// Convert AIG to Comb pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertAIGToCombPass
    : public impl::ConvertAIGToCombBase<ConvertAIGToCombPass> {

  void runOnOperation() override;
  using ConvertAIGToCombBase<ConvertAIGToCombPass>::ConvertAIGToCombBase;
};
} // namespace

static void populateAIGToCombConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AIGAndInverterOpConversion>(patterns.getContext());
}

void ConvertAIGToCombPass::runOnOperation() {
  if (!getOperation().getModuleName().starts_with("SiFive_") ||
      getOperation().getNumOutputPorts() == 0)
    return markAllAnalysesPreserved();
  ConversionTarget target(getContext());
  target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
  target.addIllegalDialect<aig::AIGDialect>();

  RewritePatternSet patterns(&getContext());
  populateAIGToCombConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
