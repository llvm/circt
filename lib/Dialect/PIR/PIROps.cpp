//===- PIROps.cpp ==-------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/PIR/PIROps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/PIR/PIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace pir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// AssertLike Canonicalizations
//===----------------------------------------------------------------------===//

namespace {
/// Remove enable if trivial, i.e. if enable == true
template <typename Op>
struct RemoveEnableTrue : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value enable = op.getEnable();
    if (!enable)
      return failure();
    auto enableConst = enable.getDefiningOp<hw::ConstantOp>();
    if (!enableConst || !enableConst.getValue().isOne())
      return failure();

    rewriter.modifyOpInPlace(op, [&]() { op.getEnableMutable().clear(); });
    return success();
  }
};

/// Delete operation if enable is `false`.
template <typename Op>
struct EraseIfEnableFalse : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value enable = op.getEnable();
    if (!enable)
      return failure();
    auto enableConst = enable.getDefiningOp<hw::ConstantOp>();
    if (!enableConst || !enableConst.getValue().isZero())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// Remove disable if trivial, i.e. if disable == false
template <typename Op>
struct RemoveDisableFalse : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value disable = op.getDisable();
    if (!disable)
      return failure();
    auto disableConst = disable.getDefiningOp<hw::ConstantOp>();
    if (!disableConst || !disableConst.getValue().isZero())
      return failure();

    rewriter.modifyOpInPlace(op, [&]() { op.getDisableMutable().clear(); });
    return success();
  }
};

/// Delete operation if disable is `true`.
template <typename Op>
struct EraseIfDisableTrue : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value disable = op.getDisable();
    if (!disable)
      return failure();
    auto disableConst = disable.getDefiningOp<hw::ConstantOp>();
    if (!disableConst || !disableConst.getValue().isOne())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// Delete operation if property is trivial, i.e. property is either
/// (bool-to-clk-prop true) or
/// (clk-seq-to-clk-prop (bool-to-clk-seq true))
template <typename Op>
struct EraseIfPropertyTrivial : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value property = op.getProperty();

    // Check for pir.bool_to_clocked_prop true
    if (auto boolToProp =
            property.template getDefiningOp<BoolToClockedPropOp>()) {
      if (auto constOp =
              boolToProp.getInput().getDefiningOp<hw::ConstantOp>()) {
        if (constOp.getValue().isOne()) {
          rewriter.eraseOp(op);
          return success();
        }
      }
    }

    // Check for pir.clocked_seq_to_clocked_prop pir.bool_to_clocked_seq true
    if (auto clkSeqToClkProp =
            property.template getDefiningOp<ClockedSeqToClockedPropOp>()) {
      if (auto boolToClkSeq =
              clkSeqToClkProp.getInput().getDefiningOp<BoolToClockedSeqOp>()) {
        if (auto constOp =
                boolToClkSeq.getInput().getDefiningOp<hw::ConstantOp>()) {
          if (constOp.getValue().isOne()) {
            rewriter.eraseOp(op);
            return success();
          }
        }
      }
    }

    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

void AssertPropertyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<EraseIfEnableFalse<AssertPropertyOp>,
              EraseIfPropertyTrivial<AssertPropertyOp>,
              EraseIfDisableTrue<AssertPropertyOp>,
              RemoveEnableTrue<AssertPropertyOp>,
              RemoveDisableFalse<AssertPropertyOp>>(context);
}

void AssumePropertyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<EraseIfEnableFalse<AssumePropertyOp>,
              EraseIfPropertyTrivial<AssumePropertyOp>,
              EraseIfDisableTrue<AssumePropertyOp>,
              RemoveEnableTrue<AssumePropertyOp>,
              RemoveDisableFalse<AssumePropertyOp>>(context);
}

void RestrictOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results
      .add<EraseIfEnableFalse<RestrictOp>, EraseIfPropertyTrivial<RestrictOp>,
           EraseIfDisableTrue<RestrictOp>, RemoveEnableTrue<RestrictOp>,
           RemoveDisableFalse<RestrictOp>>(context);
}

void CoverPropertyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<EraseIfEnableFalse<CoverPropertyOp>,
              EraseIfPropertyTrivial<CoverPropertyOp>,
              EraseIfDisableTrue<CoverPropertyOp>,
              RemoveEnableTrue<CoverPropertyOp>,
              RemoveDisableFalse<CoverPropertyOp>>(context);
}

//===----------------------------------------------------------------------===//
// Type Conversion Ops
//===----------------------------------------------------------------------===//

/// (clk-seq-to-clk-prop (bool-to-clk-seq b)) ==> (bool-to-clk-prop b)
LogicalResult
ClockedSeqToClockedPropOp::canonicalize(ClockedSeqToClockedPropOp op,
                                        PatternRewriter &rewriter) {
  if (auto bToClkSeq = op.getInput().getDefiningOp<BoolToClockedSeqOp>()) {
    rewriter.replaceOpWithNewOp<BoolToClockedPropOp>(op, bToClkSeq.getInput());
    return success();
  }
  return failure();
}

#define GET_OP_CLASSES
#include "circt/Dialect/PIR/PIR.cpp.inc"
