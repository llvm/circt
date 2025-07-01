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
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/KnownBits.h"
#include <algorithm>

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

    APInt value;

    // compress(..., 0) -> compress(...) -- identity
    if (matchPattern(inputs.back(), m_ConstantInt(&value)) && value.isZero()) {
      auto newCompressOp =
          rewriter.create<CompressOp>(op.getLoc(), inputs.drop_back(), 2);

      rewriter.replaceOp(op, newCompressOp.getResults());
      return success();
    }

    return failure();
  }
};

struct ConstantFoldPartialProduct : public OpRewritePattern<PartialProductOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PartialProductOp op,
                                PatternRewriter &rewriter) const override {
    auto operands = op.getOperands();

    auto inputType = operands[0].getType();
    unsigned inputWidth = inputType.getIntOrFloatBitWidth();

    // TODO: implement a constant folding for the PartialProductOp

    size_t maxNonZeroBits = 0;
    // pp(concat(0,a), concat(0,b)) -> reduce number of results
    for (Value operand : operands) {
      // If the extracted bits are all known, then return the result.
      auto knownBits = comb::computeKnownBits(operand);
      if (knownBits.isUnknown())
        return failure(); // Skip if we don't know anything about the bits

      size_t nonZeroBits = inputWidth - knownBits.Zero.countLeadingOnes();
      maxNonZeroBits = std::max(maxNonZeroBits, nonZeroBits);
    }

    if (maxNonZeroBits == op.getNumResults())
      return failure();

    auto newPP = rewriter.create<datapath::PartialProductOp>(
        op.getLoc(), op.getOperands(), maxNonZeroBits);

    auto zero = rewriter.create<hw::ConstantOp>(op.getLoc(),
                                                APInt::getZero(inputWidth));

    // Collect newPP results and pad with zeros if needed
    SmallVector<Value> newResults(newPP.getResults().begin(),
                                  newPP.getResults().end());
    while (newResults.size() < op.getNumResults())
      newResults.push_back(zero);

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

void CompressOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  // Add the fold pattern
  results.add<FoldAddIntoCompress, ConstantFoldCompress,
              ConstantFoldPartialProduct>(context);
}
