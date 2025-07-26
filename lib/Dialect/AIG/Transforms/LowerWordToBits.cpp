//===- LowerWordToBits.cpp - Bit-Blasting Words to Bits ---------*- C++ -*-===//
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {

struct WordRewritePattern : public OpRewritePattern<AndInverterOp> {
  using OpRewritePattern<AndInverterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AndInverterOp op,
                                PatternRewriter &rewriter) const override {
    auto width = op.getType().getIntOrFloatBitWidth();
    if (width <= 1)
      return failure();

    SmallVector<Value> results;
    // We iterate over the width in reverse order to match the endianness of
    // `comb.concat`.
    for (int64_t i = width - 1; i >= 0; --i) {
      SmallVector<Value> operands;
      for (auto operand : op.getOperands()) {
        // Reuse bits if we can extract from `comb.concat` operands.
        if (auto concat = operand.getDefiningOp<comb::ConcatOp>()) {
          // For the simplicity, we only handle the case where all the
          // `comb.concat` operands are single-bit.
          if (concat.getNumOperands() == width &&
              llvm::all_of(concat.getOperandTypes(), [](Type type) {
                return type.getIntOrFloatBitWidth() == 1;
              })) {
            // Be careful with the endianness here.
            operands.push_back(concat.getOperand(width - i - 1));
            continue;
          }
        }
        // Otherwise, we need to extract the bit.
        operands.push_back(
            comb::ExtractOp::create(rewriter, op.getLoc(), operand, i, 1));
      }
      results.push_back(AndInverterOp::create(rewriter, op.getLoc(), operands,
                                              op.getInvertedAttr()));
    }

    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, results);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower Word to Bits pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerWordToBitsPass
    : public impl::LowerWordToBitsBase<LowerWordToBitsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerWordToBitsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<WordRewritePattern>(&getContext());

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  mlir::GreedyRewriteConfig config;
  // Use top-down traversal to reuse bits from `comb.concat`.
  config.setUseTopDownTraversal(true);

  if (failed(
          mlir::applyPatternsGreedily(getOperation(), frozenPatterns, config)))
    return signalPassFailure();
}
