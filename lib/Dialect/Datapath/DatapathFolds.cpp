//===- DatapathOps.cpp - Implement the Datapath operations
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements datapath ops.
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace datapath;
using namespace matchers;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

struct FoldAddIntoCompress : public OpRewritePattern<comb::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (addOp.getNumOperands() <= 2)
      return failure(); // Not enough operands to fold

    // Get operands of the AddOp
    auto operands = addOp.getOperands();
    SmallVector<Value, 8> processedCompressorResults;
    SmallVector<Value, 8> newCompressOperands;
    auto numCompress = 0;
    bool hasRegArgs = false;

    // Check if any operand is a CompressOp and collect all operands
    for (Value operand : operands) {
      if (llvm::is_contained(processedCompressorResults, operand))
        continue; // Skip if already processed this compressor

      if (auto compressOp = operand.getDefiningOp<datapath::CompressOp>()) {
        // Found a compress op - add its operands to our new list
        ++numCompress;
        llvm::append_range(newCompressOperands, compressOp.getOperands());
        // Only process each compressor once
        llvm::append_range(processedCompressorResults, compressOp.getResults());
      } else {
        hasRegArgs = true;
        // Regular operand - just add it to our list
        newCompressOperands.push_back(operand);
      }
    }

    // If no compress was found, this pattern doesn't apply
    if (!((numCompress > 1) | ((numCompress == 1) & hasRegArgs)))
      return failure();

    // Create a new CompressOp with all collected operands
    auto newCompressOp = rewriter.create<datapath::CompressOp>(
        addOp.getLoc(), newCompressOperands, 2);

    // Replace the original AddOp with our new CompressOp
    rewriter.replaceOpWithNewOp<comb::AddOp>(addOp, newCompressOp.getResults(),
                                             true);
    return success();
  }
};

struct ConstantFoldCompress : public OpRewritePattern<CompressOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompressOp op,
                                PatternRewriter &rewriter) const override {
  auto inputs = op.getInputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  APInt value, value2;

  // compress(..., 0) -> compress(...) -- identity
  if (matchPattern(inputs.back(), m_ConstantInt(&value)) && value.isZero()) {
    auto newCompressOp = rewriter.create<CompressOp>(op.getLoc(), inputs.drop_back(), 2);

    rewriter.replaceOp(op, newCompressOp.getResults());
    return success();
  }

  return failure();
}
};

// In your DatapathOps.cpp file
void CompressOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  // Add the fold pattern
  results.add<FoldAddIntoCompress, ConstantFoldCompress>(context);
}
