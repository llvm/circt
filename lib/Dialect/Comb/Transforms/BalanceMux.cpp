//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BalanceMux pass, which balances and optimizes mux
// chains.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_BALANCEMUX
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {

/// Mux chain with comparison folding pattern.
class MuxChainWithComparison : public OpRewritePattern<MuxOp> {
  unsigned muxChainThreshold;

public:
  // Set a higher benefit than PriorityEncoderReshape to run first.
  MuxChainWithComparison(MLIRContext *context, unsigned muxChainThreshold)
      : OpRewritePattern<MuxOp>(context, /*benefit=*/2),
        muxChainThreshold(muxChainThreshold) {}
  LogicalResult matchAndRewrite(MuxOp rootMux,
                                PatternRewriter &rewriter) const override {
    auto fn = [muxChainThreshold = muxChainThreshold](size_t indexWidth,
                                                      size_t numEntries) {
      // In this pattern, we consider it beneficial to fold mux chains
      // with more than the threshold.
      if (numEntries >= muxChainThreshold)
        return MuxChainWithComparisonFoldingStyle::BalancedMuxTree;
      return MuxChainWithComparisonFoldingStyle::None;
    };
    // Try folding on both false and true sides
    return llvm::success(foldMuxChainWithComparison(rewriter, rootMux,
                                                    /*isFalseSide=*/true, fn) ||
                         foldMuxChainWithComparison(rewriter, rootMux,
                                                    /*isFalseSide=*/false, fn));
  }
};

/// Rebalances a linear chain of muxes forming a priority encoder into a
/// balanced tree structure. This reduces the depth of the mux tree from O(n)
/// to O(log n).
///
/// For a priority encoder with n conditions, this transform:
/// - Reduces depth from O(n) to O(log n) levels
/// - Muxes: Creates exactly (n-1) muxes (same as original linear chain)
/// - OR gates: Creates additional O(n log n) OR gates to combine
class PriorityMuxReshape : public OpRewritePattern<MuxOp> {
  unsigned muxChainThreshold;

public:
  PriorityMuxReshape(MLIRContext *context, unsigned muxChainThreshold)
      : OpRewritePattern<MuxOp>(context, /*benefit=*/1),
        muxChainThreshold(muxChainThreshold) {}

  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override;

private:
  /// Helper function to collect a mux chain from a given side
  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Location>>
  collectChain(MuxOp op, bool isFalseSide) const;

  /// Build a balanced tree from the collected conditions and results
  Value buildBalancedPriorityMux(PatternRewriter &rewriter,
                                 ArrayRef<Value> conditions,
                                 ArrayRef<Value> results, Value defaultValue,
                                 ArrayRef<Location> locs) const;
};
}; // namespace

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

LogicalResult
PriorityMuxReshape::matchAndRewrite(MuxOp op, PatternRewriter &rewriter) const {
  // Make sure that we're not looking at the intermediate node in a mux tree.
  if (op->hasOneUse())
    if (auto userMux = dyn_cast<MuxOp>(*op->user_begin()))
      return failure();

  // Early return if both or neither side are mux chains.
  auto trueMux = op.getTrueValue().getDefiningOp<MuxOp>();
  auto falseMux = op.getFalseValue().getDefiningOp<MuxOp>();
  if ((trueMux && falseMux) || (!trueMux && !falseMux))
    return failure();
  bool useFalseSideChain = falseMux;

  auto [conditions, results, locs] = collectChain(op, useFalseSideChain);
  if (conditions.size() < muxChainThreshold)
    return failure();

  if (!useFalseSideChain) {
    // For true-side chains, we need to invert all conditions
    for (auto &cond : conditions) {
      cond = rewriter.createOrFold<comb::XorOp>(
          op.getLoc(), cond,
          rewriter.create<hw::ConstantOp>(op.getLoc(), APInt(1, 1)), true);
    }
  }

  LDBG() << "Rebalanced priority mux with " << conditions.size()
         << " conditions, using " << (useFalseSideChain ? "false" : "true")
         << "-side chain.\n";

  assert(conditions.size() + 1 == results.size() &&
         "Expected one more result than conditions");
  ArrayRef<Value> resultsRef(results);

  // Build balanced tree and replace original op
  Value balancedTree = buildBalancedPriorityMux(
      rewriter, conditions, resultsRef.drop_back(), resultsRef.back(), locs);
  replaceOpAndCopyNamehint(rewriter, op, balancedTree);
  return success();
}

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Location>>
PriorityMuxReshape::collectChain(MuxOp op, bool isFalseSide) const {
  SmallVector<Value> chainConditions, chainResults;
  DenseSet<Value> seenConditions;
  SmallVector<Location> chainLocs;

  auto chainMux = isFalseSide ? op.getFalseValue().getDefiningOp<MuxOp>()
                              : op.getTrueValue().getDefiningOp<MuxOp>();

  if (!chainMux)
    return {chainConditions, chainResults, chainLocs};

  // Helper lambdas to abstract the differences between false/true side chains
  auto getChainResult = [&](MuxOp mux) -> Value {
    return isFalseSide ? mux.getTrueValue() : mux.getFalseValue();
  };

  auto getChainNext = [&](MuxOp mux) -> Value {
    return isFalseSide ? mux.getFalseValue() : mux.getTrueValue();
  };

  auto getRootResult = [&]() -> Value {
    return isFalseSide ? op.getTrueValue() : op.getFalseValue();
  };

  // Start collecting the chain
  seenConditions.insert(op.getCond());
  chainConditions.push_back(op.getCond());
  chainResults.push_back(getRootResult());
  chainLocs.push_back(op.getLoc());

  // Walk down the chain collecting all conditions and results
  MuxOp currentMux = chainMux;
  while (currentMux) {
    // Only add unique conditions (outer muxes have priority)
    if (seenConditions.insert(currentMux.getCond()).second) {
      chainConditions.push_back(currentMux.getCond());
      chainResults.push_back(getChainResult(currentMux));
      chainLocs.push_back(currentMux.getLoc());
    }

    auto nextMux = getChainNext(currentMux).getDefiningOp<MuxOp>();
    if (!nextMux || !nextMux->hasOneUse()) {
      // Add the final default value
      chainResults.push_back(getChainNext(currentMux));
      break;
    }
    currentMux = nextMux;
  }

  return {chainConditions, chainResults, chainLocs};
}

// This function recursively constructs a balanced binary tree of muxes for a
// priority encoder. It splits the conditions and results into halves,
// combining the left half's conditions with an OR gate to select between
// the left subtree (which includes the default for that half) and the right
// subtree. This transforms a linear chain like:
//   a_0 ? r_0 : a_1 ? r_1 : ... : a_n ? r_n : default
// into a balanced tree, reducing depth from O(n) to O(log n).
// NOLINTNEXTLINE(misc-no-recursion)
Value PriorityMuxReshape::buildBalancedPriorityMux(
    PatternRewriter &rewriter, ArrayRef<Value> conditions,
    ArrayRef<Value> results, Value defaultValue,
    ArrayRef<Location> locs) const {
  size_t size = conditions.size();
  // Base cases.
  if (size == 0)
    return defaultValue;
  if (size == 1)
    return rewriter.createOrFold<MuxOp>(locs.front(), conditions.front(),
                                        results.front(), defaultValue);

  // Recursive case: split range in half. Take the ceiling to ensure the first
  // half is larger.
  unsigned mid = llvm::divideCeil(size, 2);
  assert(mid > 0);
  auto loc = rewriter.getFusedLoc(ArrayRef<Location>(locs).take_front(mid));

  // Build left and right subtrees. Use the last result as the default for the
  // left subtree to ensure correct priority encoding.
  Value leftTree = buildBalancedPriorityMux(
      rewriter, conditions.take_front(mid), results.take_front(mid),
      results.take_front(mid).back(), locs.take_front(mid));

  Value rightTree = buildBalancedPriorityMux(
      rewriter, conditions.drop_front(mid), results.drop_front(mid),
      defaultValue, locs.drop_front(mid));

  // Combine conditions from left half with OR
  Value combinedCond =
      rewriter.createOrFold<OrOp>(loc, conditions.take_front(mid), true);

  // Create mux that selects between left and right subtrees
  return rewriter.create<MuxOp>(loc, combinedCond, leftTree, rightTree);
}

/// Pass that performs enhanced mux chain optimizations
struct BalanceMuxPass : public impl::BalanceMuxBase<BalanceMuxPass> {
  using BalanceMuxBase::BalanceMuxBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);
    patterns.add<MuxChainWithComparison, PriorityMuxReshape>(context,
                                                             muxChainThreshold);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
