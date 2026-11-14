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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

/// Given a MuxOp whose condition is an ICmpOp equality comparison, extracts
/// the non-constant subject Value and the ConstantOp. Returns std::nullopt if
/// the condition is not an ICmpOp eq with at least one hw::ConstantOp operand.
static std::optional<std::pair<Value, hw::ConstantOp>>
getEqSubjectAndConstant(MuxOp mux) {
  auto icmp = mux.getCond().getDefiningOp<ICmpOp>();
  if (!icmp || icmp.getPredicate() != ICmpPredicate::eq)
    return std::nullopt;
  if (auto cst = icmp.getRhs().getDefiningOp<hw::ConstantOp>())
    return std::make_pair(icmp.getLhs(), cst);
  if (auto cst = icmp.getLhs().getDefiningOp<hw::ConstantOp>())
    return std::make_pair(icmp.getRhs(), cst);
  return std::nullopt;
}

/// Mux chain with comparison folding pattern.
class MuxChainWithComparison : public OpRewritePattern<MuxOp> {
  unsigned muxChainThreshold;
  unsigned maxDenseIndexWidth;

public:
  // Pattern benefit ordering: MuxChainWithComparison (3) >
  // BalancedMuxTreePattern (2) > PriorityMuxReshape (1). Dense bit-extract is
  // preferred when applicable, binary search tree handles wide/sparse cases,
  // priority reshape is the general fallback.
  MuxChainWithComparison(MLIRContext *context, unsigned muxChainThreshold,
                         unsigned maxDenseIndexWidth)
      : OpRewritePattern<MuxOp>(context, /*benefit=*/3),
        muxChainThreshold(muxChainThreshold),
        maxDenseIndexWidth(maxDenseIndexWidth) {}
  LogicalResult matchAndRewrite(MuxOp rootMux,
                                PatternRewriter &rewriter) const override {
    auto fn = [this](size_t indexWidth, size_t numEntries) {
      // In this pattern, we consider it beneficial to fold mux chains
      // with more than the threshold into a dense bit-extract tree only
      // when indexWidth is within the max dense threshold.
      if (numEntries >= muxChainThreshold && indexWidth <= maxDenseIndexWidth)
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

/// Represents an individual case in a linear comparison mux chain.
struct MuxCaseEntry {
  llvm::APInt caseConstant;
  mlir::Value value;
  mlir::Location loc;
};

/// Rewrites a linear chain of MuxOp with mutually exclusive ICmpOp equality
/// comparisons into a balanced binary search tree of MuxOp with median-split
/// range comparisons.
class BalancedMuxTreePattern : public OpRewritePattern<MuxOp> {
  unsigned muxChainThreshold;

public:
  BalancedMuxTreePattern(MLIRContext *context, unsigned muxChainThreshold)
      : OpRewritePattern<MuxOp>(context, /*benefit=*/2),
        muxChainThreshold(muxChainThreshold) {}

  LogicalResult matchAndRewrite(MuxOp rootMux,
                                PatternRewriter &rewriter) const override {
    // Only match root muxes (not intermediate nodes in an existing chain).
    if (isIntermediateMux(rootMux))
      return rewriter.notifyMatchFailure(rootMux, "intermediate mux node");

    Value subject = nullptr;
    SmallVector<MuxCaseEntry, 8> cases;
    Value defaultValue = nullptr;

    if (failed(collectMuxChain(rootMux, subject, cases, defaultValue)))
      return rewriter.notifyMatchFailure(rootMux,
                                         "not a linear equality chain");

    if (cases.size() < muxChainThreshold) {
      return rewriter.notifyMatchFailure(rootMux,
                                         "mux chain length below threshold");
    }

    // Sort cases by constant value to allow recursive binary partitioning.
    llvm::sort(cases, [](const MuxCaseEntry &a, const MuxCaseEntry &b) {
      return a.caseConstant.ult(b.caseConstant);
    });

    Value balancedResult = buildBalancedTree(rewriter, rootMux.getLoc(),
                                             subject, cases, defaultValue);
    if (!balancedResult)
      return failure();

    LDBG() << "Synthesized balanced binary search tree with " << cases.size()
           << " cases\n";

    replaceOpAndCopyNamehint(rewriter, rootMux, balancedResult);
    return success();
  }

private:
  /// Checks if this mux is the false-operand of an immediate parent mux with
  /// a compatible equality comparison on the same subject.
  static bool isIntermediateMux(MuxOp mux) {
    if (!mux->hasOneUse())
      return false;
    auto parentMux = dyn_cast<MuxOp>(*mux->user_begin());
    if (!parentMux)
      return false;
    if (parentMux.getFalseValue() != mux.getResult())
      return false;

    auto parentPair = getEqSubjectAndConstant(parentMux);
    if (!parentPair)
      return false;

    auto thisPair = getEqSubjectAndConstant(mux);
    if (!thisPair)
      return false;

    return parentPair->first == thisPair->first;
  }

  /// Traverses a linear chain of muxes and collects equality comparison cases.
  static LogicalResult collectMuxChain(MuxOp rootMux, Value &commonSubject,
                                       SmallVectorImpl<MuxCaseEntry> &cases,
                                       Value &defaultValue) {
    Value currentVal = rootMux.getResult();

    while (auto currentMux = currentVal.getDefiningOp<MuxOp>()) {
      auto pair = getEqSubjectAndConstant(currentMux);
      if (!pair)
        break;

      auto [subject, constOp] = *pair;

      if (!commonSubject) {
        commonSubject = subject;
      } else if (commonSubject != subject) {
        break;
      }

      if (currentMux != rootMux && !currentMux->hasOneUse())
        break;

      cases.push_back(
          {constOp.getValue(), currentMux.getTrueValue(), currentMux.getLoc()});
      currentVal = currentMux.getFalseValue();
    }

    if (cases.empty())
      return failure();

    defaultValue = currentVal;
    return success();
  }

  /// Recursively constructs a balanced binary tree of MuxOp.
  static Value buildBalancedTree(PatternRewriter &rewriter, Location loc,
                                 Value subject,
                                 ArrayRef<MuxCaseEntry> caseSlice,
                                 Value defaultValue) {
    if (caseSlice.empty())
      return defaultValue;

    if (caseSlice.size() == 1) {
      const MuxCaseEntry &entry = caseSlice.front();
      auto caseConst =
          hw::ConstantOp::create(rewriter, entry.loc, entry.caseConstant);
      auto cond = ICmpOp::create(rewriter, entry.loc, ICmpPredicate::eq,
                                 subject, caseConst);
      return MuxOp::create(rewriter, entry.loc, cond, entry.value,
                           defaultValue);
    }

    size_t mid = caseSlice.size() / 2;
    ArrayRef<MuxCaseEntry> leftSlice = caseSlice.take_front(mid);
    ArrayRef<MuxCaseEntry> rightSlice = caseSlice.drop_front(mid);

    const llvm::APInt &splitConstVal = rightSlice.front().caseConstant;
    auto splitConst = hw::ConstantOp::create(rewriter, loc, splitConstVal);
    auto isRightCond =
        ICmpOp::create(rewriter, loc, ICmpPredicate::uge, subject, splitConst);

    Value leftTree =
        buildBalancedTree(rewriter, loc, subject, leftSlice, defaultValue);
    Value rightTree =
        buildBalancedTree(rewriter, loc, subject, rightSlice, defaultValue);

    return MuxOp::create(rewriter, loc, isRightCond, rightTree, leftTree);
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
} // namespace

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
          hw::ConstantOp::create(rewriter, op.getLoc(), APInt(1, 1)), true);
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
  // TODO: Ideally the separator index should be selected based on arrival times
  //       of results.
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
  return MuxOp::create(rewriter, loc, combinedCond, leftTree, rightTree);
}

// or(mux(c_0, a_0, 0), mux(c_1, a_1, 0), ..., mux(c_n, a_n, 0)) ->
//   mux(c_0, a_0, mux(c_1, a_1, ...)) iff all conditions are independent.
//
// The mux tree should then be balanced by other MuxOp patterns.
struct OrOfMuxToMuxChain : public OpRewritePattern<OrOp> {
  using OpRewritePattern<OrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OrOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getTwoState())
      return failure();

    SmallVector<Value> conditions;
    for (Value operand : op.getOperands()) {
      MuxOp mux = operand.getDefiningOp<MuxOp>();
      if (!mux || !mux.getTwoState() ||
          !matchPattern(mux.getFalseValue(), mlir::m_Zero()))
        return failure();

      conditions.push_back(mux.getCond());
    }

    if (!areConditionsIndependent(conditions))
      return failure();

    // Construct the mux chain from back to front.
    SmallVector<Value> values(op.getOperands().drop_back());
    Value v = op.getOperands().back();
    while (!values.empty()) {
      auto mux = values.pop_back_val().getDefiningOp<comb::MuxOp>();
      v = comb::MuxOp::create(rewriter, op.getLoc(), mux.getCond(),
                              mux.getTrueValue(), v, /*twoState=*/true);
    }
    replaceOpAndCopyNamehint(rewriter, op, v);
    return success();
  }

  // Returns true if the given i1 conditions are independent. That is, at most
  // one condition can be true at a time.
  //
  // Currently we take a shortcut and check if the conditions are defined by
  // ICmpEqs for different values. This is a common pattern in priority
  // encoders.
  bool areConditionsIndependent(ArrayRef<Value> conditions) const {
    DenseSet<IntegerAttr> seenConstants;
    for (Value v : conditions) {
      auto icmp = v.getDefiningOp<ICmpOp>();
      IntegerAttr value;
      if (!icmp || icmp.getPredicate() != ICmpPredicate::eq ||
          !matchPattern(icmp.getRhs(), mlir::m_Constant(&value)))
        return false;

      if (!seenConstants.insert(value).second)
        return false;
    }
    return true;
  }
};

/// Pass that performs enhanced mux chain optimizations
struct BalanceMuxPass : public impl::BalanceMuxBase<BalanceMuxPass> {
  using BalanceMuxBase::BalanceMuxBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);
    patterns.add<MuxChainWithComparison>(context, muxChainThreshold,
                                         maxDenseIndexWidth);
    patterns.add<BalancedMuxTreePattern>(context, muxChainThreshold);
    patterns.add<PriorityMuxReshape>(context, muxChainThreshold);
    patterns.add<OrOfMuxToMuxChain>(context);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
