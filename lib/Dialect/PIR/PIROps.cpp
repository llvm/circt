//===- PIROps.cpp ==-------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/PIR/PIROps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/PIR/PIREnums.h"
#include "circt/Dialect/PIR/PIRTypes.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace pir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/PIR/PIR.cpp.inc"

//===----------------------------------------------------------------------===//
// InputOp
//===----------------------------------------------------------------------===//

/// Attribute every name in input to its related result
void InputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Retrieve the names array
  auto namesAttr = (*this)->getAttrOfType<ArrayAttr>("names");

  // Check if any names were given
  if (!namesAttr.empty()) {
    auto results = getResults();

    // Match every name with its associated result
    for (size_t i = 0; i < results.size(); ++i) {
      // Check for string attributes in the array
      if (auto name = dyn_cast<StringAttr>(namesAttr[i])) {
        setNameFn(results[i], name);
      } else {
        // This should be unreachable, but in case raise an error
        (*this)->emitOpError("Non-string attribute in string array!");
      }
    }
  }
}

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
    // Check if an enable exists
    Value enable = op.getEnable();
    if (!enable)
      return failure();

    // Only modify if the enable is `hw.constant true`
    auto enableConst = enable.getDefiningOp<hw::ConstantOp>();
    if (!enableConst || !enableConst.getValue().isOne())
      return failure();

    // Erase enable signal
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
    // Check if there is an enable signal
    Value enable = op.getEnable();
    if (!enable)
      return failure();

    // Only modify is enable is `hw.constant false`
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
    // Check for a disable signal
    Value disable = op.getDisable();
    if (!disable)
      return failure();

    // Only modify is disable is `hw.constant false`
    auto disableConst = disable.getDefiningOp<hw::ConstantOp>();
    if (!disableConst || !disableConst.getValue().isZero())
      return failure();

    // Erase the disable input since it's useless
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
    // Do nothing if no disable was given
    if (!disable)
      return failure();

    // Skip any case where the disable is not `hw.constant true`
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

    // Check for (pir.bool_to_clocked_prop (hw.constant true))
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

    // Check for (pir.clocked_seq_to_clocked_prop
    //              (pir.bool_to_clocked_seq
    //                (hw.constant true)))
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
