//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
#define GEN_PASS_DEF_ENSUREDIVISIONISSPECULATABLE
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Rewrite DivUOp/ModUOp to ensure the divisor is not zero.
template <typename OpT>
class EnsureSpeculatablePatternU : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    Value dividend = op.getLhs();
    Value divisor = op.getRhs();

    if (isDivisorSafe(op))
      // This division is known to be safe, we do not need to do
      // anything.
      return failure();

    // Rewrite: divu(a, b) ~>
    //    pred = (b == 0)
    //    rhs_safe = pred ? 1 : b;
    //    c = divu(a, rhs_safe)
    //    result = pred ? 0 : c;
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value zero =
        hw::ConstantOp::create(b, rewriter.getIntegerAttr(op.getType(), 0));
    Value one =
        hw::ConstantOp::create(b, rewriter.getIntegerAttr(op.getType(), 1));
    Value divisorIsZero =
        comb::ICmpOp::create(b, comb::ICmpPredicate::eq, divisor, zero);
    Value safeDivisor = comb::MuxOp::create(b, divisorIsZero, one, divisor);
    auto newOp = OpT::create(b, dividend, safeDivisor);
    // Add an attribute so we don't apply recursively.
    newOp->setAttr("comb.speculatable", UnitAttr::get(b.getContext()));

    Value result = comb::MuxOp::create(b, divisorIsZero, zero, newOp);
    rewriter.replaceOp(op, result);

    return success();
  }

  bool isDivisorSafe(OpT op) const {
    // If we've already marked the divisor as safe, we don't need to do
    // anything.
    if (op->hasAttr("comb.speculatable")) {
      return true;
    }
    // Division by a constant that is nonzero is safe.
    Value divisor = op.getRhs();
    APInt v;
    if (matchPattern(divisor, m_ConstantInt(&v)) && !v.isZero()) {
      return true;
    }
    return false;
  }
};

/// Rewrite DivSOp/ModSOp to ensure the divisor is not zero and that we don't
/// calculate INT_MIN / -1, which causes overflow.
template <typename OpT>
class EnsureSpeculatablePatternS : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    Value dividend = op.getLhs();
    Value divisor = op.getRhs();
    Type ty = op.getType();

    if (isDivisorSafe(op))
      // This division is known to be safe, we do not need to do
      // anything.
      return failure();

    // Rewrite: divs(a, b) ~>
    //    pred = (b == 0 || (b == -1 && a == INT_MIN))
    //    rhs_safe = pred ? 1 : b;
    //    c = divs(a, rhs_safe)
    //    result = pred ? 0 : c;
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto eq = [&](Value lhs, Value rhs) {
      return comb::ICmpOp::create(b, comb::ICmpPredicate::eq, lhs, rhs);
    };
    auto and_ = [&](Value lhs, Value rhs) {
      return comb::AndOp::create(b, lhs, rhs);
    };
    auto or_ = [&](Value lhs, Value rhs) {
      return comb::OrOp::create(b, lhs, rhs);
    };

    int bitwidth = ty.getIntOrFloatBitWidth();
    Value zero = hw::ConstantOp::create(b, rewriter.getIntegerAttr(ty, 0));
    Value one = hw::ConstantOp::create(b, rewriter.getIntegerAttr(ty, 1));
    Value int_min = hw::ConstantOp::create(
        b, rewriter.getIntegerAttr(ty, APInt::getSignedMinValue(bitwidth)));
    Value minus_one = hw::ConstantOp::create(
        b, rewriter.getIntegerAttr(ty, APInt::getAllOnes(bitwidth)));

    Value pred = or_(eq(divisor, zero),
                     and_(eq(dividend, int_min), eq(divisor, minus_one)));
    Value safeDivisor = comb::MuxOp::create(b, pred, one, divisor);
    auto newOp = OpT::create(b, dividend, safeDivisor);
    // Add an attribute so we don't apply recursively.
    newOp->setAttr("comb.speculatable", UnitAttr::get(b.getContext()));

    Value result = comb::MuxOp::create(b, pred, zero, newOp);
    rewriter.replaceOp(op, result);
    return success();
  }

  bool isDivisorSafe(OpT op) const {
    // If we've already marked the divisor as safe, we don't need to do
    // anything.
    if (op->hasAttr("comb.speculatable")) {
      return true;
    }
    // Division by a constant that is nonzero and not minus one is safe.
    Value divisor = op.getRhs();
    APInt v;
    if (matchPattern(divisor, m_ConstantInt(&v)) && !v.isZero() &&
        !v.isAllOnes()) {
      return true;
    }
    return false;
  }
};

struct EnsureDivisionIsSpeculatablePass
    : public circt::impl::EnsureDivisionIsSpeculatableBase<
          EnsureDivisionIsSpeculatablePass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<EnsureSpeculatablePatternU<comb::DivUOp>,
                 EnsureSpeculatablePatternU<comb::ModUOp>,
                 EnsureSpeculatablePatternS<comb::DivSOp>,
                 EnsureSpeculatablePatternS<comb::ModSOp>>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::createEnsureDivisionIsSpeculatablePass() {
  return std::make_unique<EnsureDivisionIsSpeculatablePass>();
}
