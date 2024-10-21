//===- LowerWordToBits.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers multi-bit AIG operations to single-bit ones.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aig-lower-word-to-bits"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_LOWERWORDTOBITS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct WordConversion : OpConversionPattern<AndInverterOp> {
  using OpConversionPattern<AndInverterOp>::OpConversionPattern;
  using OpAdaptor = typename AndInverterOp::Adaptor;
  LogicalResult
  matchAndRewrite(AndInverterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getType().getIntOrFloatBitWidth() <= 1)
      return failure();

    SmallVector<Value> result;
    for (size_t i = 0; i < op.getType().getIntOrFloatBitWidth(); i++) {
      SmallVector<Value> operands;
      for (auto operand : op.getOperands()) {
        auto operandBits =
            rewriter.create<comb::ExtractOp>(op.getLoc(), operand, i, 1);
        operands.push_back(operandBits);
      }
      auto andInverter = rewriter.create<AndInverterOp>(op.getLoc(), operands,
                                                        op.getInvertedAttr());
      result.push_back(andInverter);
    }

    auto concat = rewriter.create<comb::ConcatOp>(op.getLoc(), result);
    rewriter.replaceOp(op, concat);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to AIG pass
//===----------------------------------------------------------------------===//

static void populateLowerWordToBitsPatterns(RewritePatternSet &patterns) {
  patterns.add<WordConversion>(patterns.getContext());
}

namespace {
struct LowerWordToBitsPass
    : public impl::LowerWordToBitsBase<LowerWordToBitsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerWordToBitsPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<comb::CombDialect>();
  target.addIllegalOp<aig::AndInverterOp>();
  target.addDynamicallyLegalOp<aig::AndInverterOp>([](AndInverterOp op) {
    return op.getType().getIntOrFloatBitWidth() <= 1;
  });

  RewritePatternSet patterns(&getContext());
  populateLowerWordToBitsPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> aig::createLowerWordToBitsPass() {
  return std::make_unique<LowerWordToBitsPass>();
}
