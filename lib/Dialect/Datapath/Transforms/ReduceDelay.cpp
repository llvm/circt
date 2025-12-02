//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/Datapath/DatapathPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace datapath {
#define GEN_PASS_DEF_DATAPATHREDUCEDELAY
#include "circt/Dialect/Datapath/DatapathPasses.h.inc"
} // namespace datapath
} // namespace circt

using namespace circt;
using namespace datapath;
using namespace mlir;

namespace {

// Fold add operations even if used multiple times incurring area overhead as
// transformation reduces shared logic - but reduces delay
// add = a + b;
// out1 = add + c;
// out2 = add << d;
// -->
// add = a + b;
// comp1 = compress(a, b, c);
// out1 = comp1[0] + comp1[1];
// out2 = add << d;
struct FoldAddReplicate : public OpRewritePattern<comb::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AddOp addOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 8> newCompressOperands;
    // Check if any operand is an AddOp that has not be folded by comb folds
    for (Value operand : addOp.getOperands()) {
      if (comb::AddOp nestedAddOp = operand.getDefiningOp<comb::AddOp>()) {
        llvm::append_range(newCompressOperands, nestedAddOp.getOperands());
      } else {
        newCompressOperands.push_back(operand);
      }
    }

    // Nothing to be folded
    if (newCompressOperands.size() <= addOp.getNumOperands())
      return failure();

    // Create a new CompressOp with all collected operands
    auto newCompressOp = datapath::CompressOp::create(rewriter, addOp.getLoc(),
                                                      newCompressOperands, 2);

    // Add the results of the CompressOp
    rewriter.replaceOpWithNewOp<comb::AddOp>(addOp, newCompressOp.getResults(),
                                             true);
    return success();
  }
};

// (a ? b + c : d + e) + f
// -->
// (a ? b : d) + (a ? c : e) + f
struct FoldMuxAdd : public OpRewritePattern<comb::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  // When used in conjunction with datapath canonicalization will only replicate
  // two input adders.
  LogicalResult matchAndRewrite(comb::AddOp addOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 8> newCompressOperands;
    for (Value operand : addOp.getOperands()) {
      // Detect a mux operand - then check if it contains add operations
      comb::MuxOp nestedMuxOp = operand.getDefiningOp<comb::MuxOp>();

      // If not matched just add the operand without modification
      if (!nestedMuxOp) {
        newCompressOperands.push_back(operand);
        continue;
      }

      SmallVector<Value> trueValOperands = {nestedMuxOp.getTrueValue()};
      SmallVector<Value> falseValOperands = {nestedMuxOp.getFalseValue()};
      // match a ? b + c : xx
      if (comb::AddOp trueVal =
              nestedMuxOp.getTrueValue().getDefiningOp<comb::AddOp>())
        trueValOperands = trueVal.getOperands();

      // match a ? xx : c + d
      if (comb::AddOp falseVal =
              nestedMuxOp.getFalseValue().getDefiningOp<comb::AddOp>())
        falseValOperands = falseVal.getOperands();

      auto maxOperands =
          std::max(trueValOperands.size(), falseValOperands.size());

      // No nested additions
      if (maxOperands <= 1) {
        newCompressOperands.push_back(operand);
        continue;
      }

      // Pad with zeros to match number of operands
      // a ? b + c : d -> (a ? b : d) + (a ? c : 0)
      auto zero =
          hw::ConstantOp::create(rewriter, addOp.getLoc(),
                                 rewriter.getIntegerAttr(addOp.getType(), 0));
      for (size_t i = 0; i < maxOperands; ++i) {
        auto tOp = i < trueValOperands.size() ? trueValOperands[i] : zero;
        auto fOp = i < falseValOperands.size() ? falseValOperands[i] : zero;
        auto newMux = comb::MuxOp::create(rewriter, addOp.getLoc(),
                                          nestedMuxOp.getCond(), tOp, fOp);
        newCompressOperands.push_back(newMux.getResult());
      }
    }

    // Nothing to be folded
    if (newCompressOperands.size() <= addOp.getNumOperands())
      return failure();

    // Create a new CompressOp with all collected operands
    auto newCompressOp = datapath::CompressOp::create(rewriter, addOp.getLoc(),
                                                      newCompressOperands, 2);

    // Add the results of the CompressOp
    rewriter.replaceOpWithNewOp<comb::AddOp>(addOp, newCompressOp.getResults(),
                                             true);
    return success();
  }
};

struct CombICmpOpConversion : public OpRewritePattern<comb::ICmpOp> {
  using OpRewritePattern::OpRewritePattern;

  // Applicable to unsigned comparisons without overflow:
  // a + b < c + d
  // -->
  // msb( {0,a} + {0,b} - {0,c} - {0,d} )
  LogicalResult matchAndRewrite(comb::ICmpOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto width = lhs.getType().getIntOrFloatBitWidth();

    // Only unsigned comparisons
    if (op.getPredicate() != comb::ICmpPredicate::ult &&
        op.getPredicate() != comb::ICmpPredicate::ule &&
        op.getPredicate() != comb::ICmpPredicate::ugt &&
        op.getPredicate() != comb::ICmpPredicate::uge)
      return failure();

    //                                         lhsMinusRhs       invertOut
    //---------------------------------------------------------------------
    // ult: a < b -> a - b < 0                       true           false
    // uge: a > b -> b - a < 0                      false           false
    // uge: a >= b -> !(a < b) -> !(a - b < 0)       true            true
    // ule: a <= b -> !(a > b) -> !(b - a < 0)      false            true
    bool lhsMinusRhs = op.getPredicate() == comb::ICmpPredicate::ult ||
                       op.getPredicate() == comb::ICmpPredicate::uge;

    bool invertOut = op.getPredicate() == comb::ICmpPredicate::uge ||
                     op.getPredicate() == comb::ICmpPredicate::ule;

    // Compute rhs - lhs
    if (!lhsMinusRhs)
      std::swap(lhs, rhs);
    SmallVector<Value> lhsAddends = {lhs};
    // Detect adder inputs to either side of the comparison and detect overflow
    if (comb::AddOp lhsAdd = lhs.getDefiningOp<comb::AddOp>()) {
      auto overflow = lhsAdd->getAttrOfType<BoolAttr>("comb.nuw");
      if (overflow && !overflow.getValue())
        lhsAddends = lhsAdd.getOperands();
    }

    SmallVector<Value> rhsAddends = {rhs};
    // Detect adder inputs to either side of the comparison and detect overflow
    if (comb::AddOp rhsAdd = rhs.getDefiningOp<comb::AddOp>()) {
      auto overflow = rhsAdd->getAttrOfType<BoolAttr>("comb.nuw");
      if (overflow && !overflow.getValue())
        rhsAddends = rhsAdd.getOperands();
    }

    // No benefit to folding into a single addition - more expensive than
    // the original comparison
    if (lhsAddends.size() + rhsAddends.size() < 3)
      return failure();

    SmallVector<Value> lhsExtend;
    for (auto addend : lhsAddends) {
      auto ext = comb::createZExt(rewriter, op.getLoc(), addend, width + 1);
      lhsExtend.push_back(ext);
    }

    SmallVector<Value> rhsExtend;
    for (auto addend : rhsAddends) {
      auto ext = comb::createZExt(rewriter, op.getLoc(), addend, width + 1);
      auto negatedAddend = comb::createOrFoldNot(op.getLoc(), ext, rewriter);
      rhsExtend.push_back(negatedAddend);
    }

    rhsExtend.push_back(hw::ConstantOp::create(
        rewriter, op.getLoc(), APInt(width + 1, rhsExtend.size())));

    SmallVector<Value> allAddends;
    llvm::append_range(allAddends, lhsExtend);
    llvm::append_range(allAddends, rhsExtend);
    auto add = rewriter.create<comb::AddOp>(op.getLoc(), allAddends, false);
    auto msb = rewriter.createOrFold<comb::ExtractOp>(
        op.getLoc(), add.getResult(), width, 1);

    if (!invertOut) {
      rewriter.replaceOp(op, msb);
      return success();
    }

    auto notOp = comb::createOrFoldNot(op.getLoc(), msb, rewriter);
    rewriter.replaceOp(op, notOp);
    return success();
  }
};

} // namespace

namespace {
struct DatapathReduceDelayPass
    : public circt::datapath::impl::DatapathReduceDelayBase<
          DatapathReduceDelayPass> {

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FoldAddReplicate, FoldMuxAdd, CombICmpOpConversion>(ctx);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace
