//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// Compress Operation
//===----------------------------------------------------------------------===//
// Check that all compressor results are included in this list of operands
// If not we must take care as manipulating compressor results independently
// could easily introduce a non-equivalent representation.
static bool areAllCompressorResultsSummed(ValueRange compressResults,
                                          ValueRange operands) {
  for (auto result : compressResults) {
    if (!llvm::is_contained(operands, result))
      return false;
  }
  return true;
}

struct FoldCompressIntoCompress
    : public OpRewritePattern<datapath::CompressOp> {
  using OpRewritePattern::OpRewritePattern;

  // compress(compress(a,b,c), add(e,f)) -> compress(a,b,c,e,f)
  LogicalResult matchAndRewrite(datapath::CompressOp compOp,
                                PatternRewriter &rewriter) const override {
    auto operands = compOp.getOperands();
    llvm::SmallSetVector<Value, 8> processedCompressorResults;
    SmallVector<Value, 8> newCompressOperands;

    for (Value operand : operands) {

      // Skip if already processed this compressor
      if (processedCompressorResults.contains(operand))
        continue;

      // If the operand has multiple uses, we do not fold it into a compress
      // operation, so we treat it as a regular operand to maintain sharing.
      if (!operand.hasOneUse()) {
        newCompressOperands.push_back(operand);
        continue;
      }

      // Found a compress op - add its operands to our new list
      if (auto compressOp = operand.getDefiningOp<datapath::CompressOp>()) {

        // Check that all results of the compressor are summed in this add
        if (!areAllCompressorResultsSummed(compressOp.getResults(), operands))
          return failure();

        llvm::append_range(newCompressOperands, compressOp.getOperands());
        // Only process each compressor once as multiple operands will point
        // to the same defining operation
        processedCompressorResults.insert(compressOp.getResults().begin(),
                                          compressOp.getResults().end());
        continue;
      }

      if (auto addOp = operand.getDefiningOp<comb::AddOp>()) {
        llvm::append_range(newCompressOperands, addOp.getOperands());
        continue;
      }

      // Regular operand - just add it to our list
      newCompressOperands.push_back(operand);
    }

    // If unable to collect more operands then this pattern doesn't apply
    if (newCompressOperands.size() <= compOp.getNumOperands())
      return failure();

    // Create a new CompressOp with all collected operands
    rewriter.replaceOpWithNewOp<datapath::CompressOp>(
        compOp, newCompressOperands, compOp.getNumResults());
    return success();
  }
};

struct FoldAddIntoCompress : public OpRewritePattern<comb::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  // add(compress(a,b,c),d) -> add(compress(a,b,c,d))
  LogicalResult matchAndRewrite(comb::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    // comb.add canonicalization patterns handle folding add operations
    if (addOp.getNumOperands() <= 2)
      return failure();

    // Get operands of the AddOp
    auto operands = addOp.getOperands();
    llvm::SmallSetVector<Value, 8> processedCompressorResults;
    SmallVector<Value, 8> newCompressOperands;
    // Only construct compressor if can form a larger compressor than what
    // is currently an input of this add
    bool shouldFold = false;

    for (Value operand : operands) {

      // Skip if already processed this compressor
      if (processedCompressorResults.contains(operand))
        continue;

      // If the operand has multiple uses, we do not fold it into a compress
      // operation, so we treat it as a regular operand.
      if (!operand.hasOneUse()) {
        shouldFold |= !newCompressOperands.empty();
        newCompressOperands.push_back(operand);
        continue;
      }

      // Found a compress op - add its operands to our new list
      if (auto compressOp = operand.getDefiningOp<datapath::CompressOp>()) {

        // Check that all results of the compressor are summed in this add
        if (!areAllCompressorResultsSummed(compressOp.getResults(), operands))
          return failure();

        // If we've already added one operand it should be folded
        shouldFold |= !newCompressOperands.empty();
        llvm::append_range(newCompressOperands, compressOp.getOperands());
        // Only process each compressor once
        processedCompressorResults.insert(compressOp.getResults().begin(),
                                          compressOp.getResults().end());
        continue;
      }

      if (auto addOp = operand.getDefiningOp<comb::AddOp>()) {
        shouldFold |= !newCompressOperands.empty();
        llvm::append_range(newCompressOperands, addOp.getOperands());
        continue;
      }

      // Regular operand - just add it to our list
      shouldFold |= !newCompressOperands.empty();
      newCompressOperands.push_back(operand);
    }

    // Only fold if we have constructed a larger compressor than what was
    // already there
    if (!(shouldFold))
      return failure();

    // Create a new CompressOp with all collected operands
    auto newCompressOp = rewriter.create<datapath::CompressOp>(
        addOp.getLoc(), newCompressOperands, 2);

    // Replace the original AddOp with a new add(compress(inputs))
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

    APInt value;

    // compress(..., 0) -> compress(...) -- identity
    if (matchPattern(inputs.back(), m_ConstantInt(&value)) && value.isZero()) {

      // If only reducing by one row and contains zero - pass through operands
      if (size - 1 == op.getNumResults()) {
        rewriter.replaceOp(op, inputs.drop_back());
        return success();
      }

      // Default create a compressor with fewer arguments
      auto newCompressOp = rewriter.create<CompressOp>(
          op.getLoc(), inputs.drop_back(), op.getNumResults());

      rewriter.replaceOp(op, newCompressOp.getResults());
      return success();
    }

    return failure();
  }
};

void CompressOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {

  results
      .add<FoldCompressIntoCompress, FoldAddIntoCompress, ConstantFoldCompress>(
          context);
}

//===----------------------------------------------------------------------===//
// Partial Product Operation
//===----------------------------------------------------------------------===//
struct ConstantFoldPartialProduct : public OpRewritePattern<PartialProductOp> {
  using OpRewritePattern::OpRewritePattern;

  // pp(concat(0,a), concat(0,b)) -> reduced number of results
  LogicalResult matchAndRewrite(PartialProductOp op,
                                PatternRewriter &rewriter) const override {
    auto operands = op.getOperands();
    unsigned inputWidth = operands[0].getType().getIntOrFloatBitWidth();

    // TODO: implement a constant multiplication for the PartialProductOp

    size_t maxNonZeroBits = 0;
    for (Value operand : operands) {
      // If the extracted bits are all known, then return the result.
      auto knownBits = comb::computeKnownBits(operand);
      if (knownBits.isUnknown())
        return failure(); // Skip if we don't know anything about the bits

      size_t nonZeroBits = inputWidth - knownBits.Zero.countLeadingOnes();

      // If all bits non-zero we will not reduce the number of results
      if (nonZeroBits == op.getNumResults())
        return failure();

      maxNonZeroBits = std::max(maxNonZeroBits, nonZeroBits);
    }

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

void PartialProductOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {

  results.add<ConstantFoldPartialProduct>(context);
}
