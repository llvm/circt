//===- CombStrengthReduction.cpp - A Combinational Strength Reduction pass
//----------===//
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

static std::pair<size_t, size_t>
computeCommonPrefixAndSuffix(const OperandRange &a, const OperandRange &b) {
  size_t commonPrefixLength = 0;
  size_t commonSuffixLength = 0;
  size_t sizeA = a.size();
  size_t sizeB = b.size();

  for (; commonPrefixLength < std::min(sizeA, sizeB); commonPrefixLength++) {
    if (a[commonPrefixLength] != b[commonPrefixLength]) {
      break;
    }
  }

  for (; commonSuffixLength < std::min(sizeA, sizeB) - commonPrefixLength;
       commonSuffixLength++) {
    if (a[sizeA - commonSuffixLength - 1] !=
        b[sizeB - commonSuffixLength - 1]) {
      break;
    }
  }

  return {commonPrefixLength, commonSuffixLength};
}

static bool isPredicateSigned(const ICmpPredicate &predicate) {
  switch (predicate) {
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

static mlir::Optional<Value> findFirstNonEmptyValue(const OperandRange &range) {
  for (auto op : range) {
    if (op.getType().getIntOrFloatBitWidth() > 0) {
      return mlir::Optional<Value>(op);
    }
  }

  return None;
}

static size_t calculateRangeTotalWidth(const OperandRange &range) {
  size_t totalWidth = 0;
  for (auto operand : range) {
    // getIntOrFloatBitWidth should never raise, since all arguments to ConcatOp
    // are integers.
    ssize_t width = operand.getType().getIntOrFloatBitWidth();
    assert(width >= 0);
    totalWidth += width;
  }
  return totalWidth;
}

// Reduce the strength icmp(concat(...), concat(...)) by doing a element-wise
// comparison on common prefix and suffixes. Returns success() if a rewriting
// happens.
static LogicalResult matchAndRewriteCompareConcat(ICmpOp &op, ConcatOp &lhs,
                                                  ConcatOp &rhs,
                                                  PatternRewriter &rewriter) {
  auto lhsOperands = lhs.getOperands();
  auto rhsOperands = rhs.getOperands();
  size_t numElements = std::min<size_t>(lhsOperands.size(), rhsOperands.size());

  std::pair<size_t, size_t> commonPrefixSuffixLength =
      computeCommonPrefixAndSuffix(lhsOperands, rhsOperands);
  size_t commonPrefixLength = commonPrefixSuffixLength.first;
  size_t commonSuffixLength = commonPrefixSuffixLength.second;

  auto replaceWithConstantI1 = [&](bool constant) -> LogicalResult {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, APInt(1, constant));
    return success();
  };

  auto directOrCat = [&](const OperandRange &range) -> Value {
    assert(range.size() > 0);
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
    size_t commonPrefixTotalWidth =
        calculateRangeTotalWidth(lhsOperands.take_front(commonPrefixLength));
    size_t commonSuffixTotalWidth =
        calculateRangeTotalWidth(lhsOperands.take_back(commonSuffixLength));
    auto lhsOnly = lhsOperands.drop_front(commonPrefixLength)
                       .drop_back(commonSuffixLength);
    auto rhsOnly = rhsOperands.drop_front(commonPrefixLength)
                       .drop_back(commonSuffixLength);

    auto replaceWithoutReplicatingSignBit = [&]() {
      auto newLhs = directOrCat(lhsOnly);
      auto newRhs = directOrCat(rhsOnly);
      return replaceWith(op.predicate(), newLhs, newRhs);
    };

    auto replaceWithReplicatingSignBit = [&]() {
      auto firstNonEmptyValue =
          findFirstNonEmptyValue(lhs.getOperands()).getValue();
      auto firstNonEmptyElemWidth =
          firstNonEmptyValue.getType().getIntOrFloatBitWidth();
      Value signBit;

      // Skip creating an ExtractOp where possible.
      if (firstNonEmptyElemWidth == 1) {
        signBit = firstNonEmptyValue;
      } else {
        signBit = rewriter.create<ExtractOp>(
            op.getLoc(), IntegerType::get(rewriter.getContext(), 1),
            firstNonEmptyValue, firstNonEmptyElemWidth - 1);
      }

      auto newLhs = rewriter.create<ConcatOp>(op.getLoc(), signBit, lhsOnly);
      auto newRhs = rewriter.create<ConcatOp>(op.getLoc(), signBit, rhsOnly);
      return replaceWith(op.predicate(), newLhs, newRhs);
    };

    if (isPredicateSigned(op.predicate())) {

      // scmp(cat(..x, b), cat(..y, b)) == scmp(cat(..x), cat(..y))
      if (commonPrefixTotalWidth == 0 && commonSuffixTotalWidth > 0) {
        return replaceWithoutReplicatingSignBit();
      }

      // scmp(cat(a, ..x, b), cat(a, ..y, b)) == scmp(cat(sgn(a), ..x),
      // cat(sgn(b), ..y)) Note that we cannot perform this optimization if
      // [width(b) = 0 && width(a) <= 1]. since that common prefix is the sign
      // bit. Doing the rewrite can result in an infinite loop.
      if (commonPrefixTotalWidth > 1 || commonSuffixTotalWidth > 0) {
        return replaceWithReplicatingSignBit();
      }

    } else {

      // ucmp(cat(a, ..x, b), cat(a, ..y, b)) = ucmp(cat(..x), cat(..y))
      if (commonPrefixTotalWidth > 0 || commonSuffixTotalWidth > 0) {
        return replaceWithoutReplicatingSignBit();
      }
    }
  }

  return failure();
}

namespace {

class RewriteCompareConcat : public mlir::RewritePattern {
public:
  RewriteCompareConcat(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /* benefit */ 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto cmpOp = dyn_cast_or_null<ICmpOp>(op)) {
      if (auto lhs = dyn_cast_or_null<ConcatOp>(cmpOp.lhs().getDefiningOp())) {
        if (auto rhs =
                dyn_cast_or_null<ConcatOp>(cmpOp.rhs().getDefiningOp())) {
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
  (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(), patterns,
                                     config);
}

std::unique_ptr<Pass> circt::createCombStrengthReductionPass() {
  return std::make_unique<CombStrengthReduction>();
}
