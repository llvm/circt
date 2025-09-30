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
// add = a+b;
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
    auto newCompressOp = rewriter.create<datapath::CompressOp>(
        addOp.getLoc(), newCompressOperands, 2);

    // Add the results of the CompressOp
    rewriter.replaceOpWithNewOp<comb::AddOp>(addOp, newCompressOp.getResults(),
                                             true);
    return success();
  }
};

// (a ? b+c : d+e) + f
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
      if (comb::MuxOp nestedMuxOp = operand.getDefiningOp<comb::MuxOp>()) {

        SmallVector<Value> trueValOperands = {nestedMuxOp.getTrueValue()};
        SmallVector<Value> falseValOperands = {nestedMuxOp.getFalseValue()};
        // match a ? b+c : xx
        if (comb::AddOp trueVal =
                nestedMuxOp.getTrueValue().getDefiningOp<comb::AddOp>())
          trueValOperands = trueVal.getOperands();

        // match a ? xx : c+d
        if (comb::AddOp falseVal =
                nestedMuxOp.getFalseValue().getDefiningOp<comb::AddOp>())
          falseValOperands = falseVal.getOperands();

        auto maxOperands =
            std::max(trueValOperands.size(), falseValOperands.size());

        // Pad with zeros to match number of operands
        // a ? b+c : d -> (a ? b : d) + (a ? c : 0)
        if (maxOperands > 1) {
          auto zero = rewriter.create<hw::ConstantOp>(
              addOp.getLoc(), rewriter.getIntegerAttr(addOp.getType(), 0));
          for (size_t i = 0; i < maxOperands; ++i) {
            auto tOp = i < trueValOperands.size() ? trueValOperands[i] : zero;
            auto fOp = i < falseValOperands.size() ? falseValOperands[i] : zero;
            auto newMux = rewriter.create<comb::MuxOp>(
                addOp.getLoc(), nestedMuxOp.getCond(), tOp, fOp);
            newCompressOperands.push_back(newMux.getResult());
          }
          continue;
        }
      }
      // If not matched just add the operand without modification
      newCompressOperands.push_back(operand);
    }

    // Nothing to be folded
    if (newCompressOperands.size() <= addOp.getNumOperands())
      return failure();

    // Create a new CompressOp with all collected operands
    auto newCompressOp = rewriter.create<datapath::CompressOp>(
        addOp.getLoc(), newCompressOperands, 2);

    // Add the results of the CompressOp
    rewriter.replaceOpWithNewOp<comb::AddOp>(addOp, newCompressOp.getResults(),
                                             true);
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
    patterns.add<FoldAddReplicate, FoldMuxAdd>(ctx);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::Pass> circt::datapath::createDatapathReduceDelayPass() {
  return std::make_unique<DatapathReduceDelayPass>();
}
