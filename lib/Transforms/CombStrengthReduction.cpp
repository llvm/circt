//===- CombStrengthReduction.cpp - A Combinational Strength Reduction pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simplified canonicalizer pass that doesn't do CFG
// optimizations and other things that aren't helpful for many hardware IRs.
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace comb;

#define GEN_PASS_CLASSES
#include "circt/Transforms/Passes.h.inc"

namespace {

static std::pair<size_t, size_t> computeCommonPrefixAndSuffix(const OperandRange &a, const OperandRange &b)
{
  size_t commonPrefixLength = 0;
  size_t commonSuffixLength = 0;
  const size_t sizeA = a.size();
  const size_t sizeB = b.size();

  for (; commonPrefixLength < std::min(sizeA, sizeB); commonPrefixLength++) {
    if (a[commonPrefixLength] != b[commonPrefixLength]) {
      break;
    }
  }

  for (; commonSuffixLength < std::min(sizeA, sizeB) - commonPrefixLength; commonSuffixLength++) {
    if (a[sizeA - commonSuffixLength - 1] != b[sizeB - commonSuffixLength - 1]) {
      break;
    }
  }

  return { commonPrefixLength, commonSuffixLength };
}

static bool isPredicateSigned (const ICmpPredicate & predicate)
{
  switch (predicate)  {
    case ICmpPredicate::ult:
    case ICmpPredicate::ugt:
    case ICmpPredicate::ule:
    case ICmpPredicate::uge:
    case ICmpPredicate::ne:
    case ICmpPredicate::eq:
      return false;
    case ICmpPredicate::slt:
    case ICmpPredicate::sgt:
    case ICmpPredicate::sle:
    case ICmpPredicate::sge:
      return true;
  }
  llvm_unreachable("unknown comparison predicate");
}

static mlir::Optional<Value> findFirstNonEmptyValue(const OperandRange &range)
{
  for (auto op : range) {
    if (op.getType().getIntOrFloatBitWidth() > 1) {
      return mlir::Optional<Value>(op);
    }
  }

  return mlir::None;
}

LogicalResult matchAndRewriteCompareConcat(ICmpOp &op, ConcatOp &lhs, ConcatOp &rhs, PatternRewriter &rewriter) {

  auto lhsOperands = lhs.getOperands();
  auto rhsOperands = lhs.getOperands();
  size_t numElements = std::min<size_t>(lhsOperands.size(), rhsOperands.size());

  const std::pair<size_t, size_t> commonPrefixSuffixLength = computeCommonPrefixAndSuffix(
      lhs.getOperands(), rhs.getOperands());
  const size_t commonPrefixLength = commonPrefixSuffixLength.first;
  const size_t commonSuffixLength = commonPrefixSuffixLength.second;

  auto replaceWithConstantI1 = [&](bool constant) -> LogicalResult {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, APInt(1, constant));
    return success();
  };

  auto directOrCat = [&](OperandRange range) -> mlir::Value {
    assert (range.size() > 0);
    if (range.size() == 1) {
      return range.front();
    }

    return rewriter.create<ConcatOp>(op.getLoc(), range);
  };

  auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
      Value rhs) -> LogicalResult {
    rewriter.replaceOpWithNewOp<ICmpOp>(op, predicate, lhs, rhs);
    return success();
  };

  if (commonPrefixLength == numElements) {
    // cat(a, b, c) == cat(a, b, c) -> 1
    switch (op.predicate()) {
      case ICmpPredicate::ult:
      case ICmpPredicate::ugt:
      case ICmpPredicate::ne:
      case ICmpPredicate::slt:
      case ICmpPredicate::sgt:
        return replaceWithConstantI1(0);

      case ICmpPredicate::sle:
      case ICmpPredicate::sge:
      case ICmpPredicate::ule:
      case ICmpPredicate::uge:
      case ICmpPredicate::eq:
        return replaceWithConstantI1(1);
    }
  } else {
    bool shouldAttemptRewrite = (commonSuffixLength > 0);

    if (!shouldAttemptRewrite && commonPrefixLength > 0) {
      size_t totalWidth = 0;

      // In the absence of common suffix, common prefix of length 1 with the sign bit
      // is already simplfied, and should not be rewritten to avoid infinite loops.
      for (const Value operand : lhs.getOperands().take_front(commonPrefixLength)) {
        totalWidth += operand.getType().getIntOrFloatBitWidth();
        if (totalWidth > 1) {
          shouldAttemptRewrite = true;
          break;
        }
      }
    }

    if (shouldAttemptRewrite) {
      auto lhsOnly = lhs.getOperands().drop_front(commonPrefixLength).drop_back(commonSuffixLength);
      auto rhsOnly = rhs.getOperands().drop_front(commonPrefixLength).drop_back(commonSuffixLength);
      auto commonPrefixContainsSignBit = false;

      // this is a stronger, but necessary requirement than that of
      // [commonPrefixContainsSignBit = commonPrefixLength > 0] to account for concat arguments
      // with zero-length widths.
      for (size_t i = 0; i < commonPrefixLength ; i++) {
        if (lhs.getOperand(i).getType().getIntOrFloatBitWidth() > 0) {
          commonPrefixContainsSignBit = true;
          break;
        }
      }

      if (!isPredicateSigned(op.predicate()) || !commonPrefixContainsSignBit) {
        const auto newLhs = directOrCat(lhsOnly);
        const auto newRhs = directOrCat(rhsOnly);
        return replaceWith(op.predicate(), newLhs, newRhs);
      } 

      auto firstNonEmptyValue = findFirstNonEmptyValue(lhs.getOperands()).getValue();
      auto signBit = rewriter.create<ExtractOp>(
          op.getLoc(),
          IntegerType::get(rewriter.getContext(), 1),
          firstNonEmptyValue,
          firstNonEmptyValue.getType().getIntOrFloatBitWidth() - 1);
      auto newLhs = rewriter.create<ConcatOp>(op.getLoc(), signBit, lhsOnly);
      auto newRhs = rewriter.create<ConcatOp>(op.getLoc(), signBit, rhsOnly);
      return replaceWith(op.predicate(), newLhs, newRhs);
    }
  }

  return failure();
}

class RewriteCompareConcat : public mlir::RewritePattern {
public:
  RewriteCompareConcat(MLIRContext *context) : RewritePattern(
      MatchAnyOpTypeTag(), /* benefit */ 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // TODO: This elimination can be done more aggressively on non-concat operations, for example:
    // cmp(a[31:0], concat(a[31:16], b[15:0])) could be put through this optimization.

    // icmp (cat(x, a, y), cat(x, b, y)) -> icmp(a, b) with special handling for sign extensions.
    if (auto cmpOp = dyn_cast_or_null<ICmpOp>(op)) {
      if (auto lhs = dyn_cast_or_null<ConcatOp>(cmpOp.lhs().getDefiningOp())) {
        if (auto rhs = dyn_cast_or_null<ConcatOp>(cmpOp.rhs().getDefiningOp())) {
          return matchAndRewriteCompareConcat(cmpOp, lhs, rhs, rewriter);
        }
      }
    }

    return failure();
  }
};

class CombStrengthReduction
  : public CombStrengthReductionBase<CombStrengthReduction> {

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    owningPatterns.add<RewriteCompareConcat>(context);
    patterns = std::move(owningPatterns);
    return success();
  };

  void runOnOperation() override;

  mlir::FrozenRewritePatternSet patterns;
};
} // end anonymous namespace

void CombStrengthReduction::runOnOperation() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  (void) applyPatternsAndFoldGreedily(getOperation()->getRegions(), patterns, config);
}

std::unique_ptr<Pass> circt::createCombStrengthReductionPass() {
  return std::make_unique<CombStrengthReduction>();
}
