//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTINDEXTOUINT
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Rewrite `arith.cmpi` operations that still reason about `index` values into
/// pure integer comparisons so that subsequent hardware mappings only observe
/// integer arithmetic.
class IndexCmpToIntegerPattern : public OpRewritePattern<arith::CmpIOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getLhs().getType().isIndex() || !op.getRhs().getType().isIndex())
      return failure();

    FailureOr<IntegerType> targetType = getTargetIntegerType(op);
    if (failed(targetType))
      return failure();

    // Peel index operands back to the original integer type: either drop an
    // index_cast (only if it came from the exact target integer type) or
    // rebuild an index constant as an integer constant. Anything else keeps
    // the pattern from firing so we never rewrite mixed or ambiguous operands.
    auto convertOperand = [&](Value operand) -> FailureOr<Value> {
      if (auto castOp = operand.getDefiningOp<arith::IndexCastOp>()) {
        Value source = castOp.getIn();
        auto srcType = dyn_cast<IntegerType>(source.getType());
        if (!srcType || srcType != *targetType)
          return failure();
        return source;
      }

      if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
        if (!constOp.getType().isIndex())
          return failure();

        auto value = dyn_cast<IntegerAttr>(constOp.getValue());
        if (!value)
          return failure();

        auto attr = rewriter.getIntegerAttr(*targetType, value.getInt());
        auto newConst =
            arith::ConstantOp::create(rewriter, constOp.getLoc(), attr);
        return newConst.getResult();
      }

      return failure();
    };

    FailureOr<Value> lhs = convertOperand(op.getLhs());
    FailureOr<Value> rhs = convertOperand(op.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, op.getPredicate(), *lhs,
                                               *rhs);
    return success();
  }

private:
  static FailureOr<IntegerType> getTargetIntegerType(arith::CmpIOp op) {
    auto pickType = [](Value operand) -> FailureOr<IntegerType> {
      if (auto castOp = operand.getDefiningOp<arith::IndexCastOp>()) {
        if (auto srcType = dyn_cast<IntegerType>(castOp.getIn().getType()))
          return srcType;
      }
      return failure();
    };

    auto lhsType = pickType(op.getLhs());
    if (succeeded(lhsType))
      return *lhsType;

    auto rhsType = pickType(op.getRhs());
    if (succeeded(rhsType))
      return *rhsType;

    return failure();
  }
};

/// Drop `arith.index_cast` that became unused once comparisons were rewritten.
class DropUnusedIndexCastPattern
    : public OpRewritePattern<arith::IndexCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Remove `arith.constant` index definitions that no longer feed any user.
class DropUnusedIndexConstantPattern
    : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isIndex() || !op->use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertIndexToUIntPass
    : public circt::impl::ConvertIndexToUIntBase<ConvertIndexToUIntPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<IndexCmpToIntegerPattern, DropUnusedIndexCastPattern,
                 DropUnusedIndexConstantPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertIndexToUIntPass() {
  return std::make_unique<ConvertIndexToUIntPass>();
}
