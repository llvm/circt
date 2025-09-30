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
struct FoldAddReplicate : public OpRewritePattern<comb::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(comb::AddOp addOp,
                                PatternRewriter &rewriter) const override {

    auto replicatedAddOps = 0;
    SmallVector<Value, 8> newCompressOperands;
    // Check if any operand is an AddOp that will not be folded by comb folds
    for (Value operand : addOp.getOperands()) {
      if (comb::AddOp nestedAddOp = operand.getDefiningOp<comb::AddOp>()) {
        llvm::append_range(newCompressOperands, nestedAddOp.getOperands());
        ++replicatedAddOps;
      } else {
        newCompressOperands.push_back(operand);
      }
    }

    if (replicatedAddOps < 1)
      return failure(); // No nested AddOp found to fold

    // Create a new CompressOp with all collected operands
    auto newCompressOp = rewriter.create<datapath::CompressOp>(
        addOp.getLoc(), newCompressOperands, 2);

    // Replace the original AddOp with our new CompressOp
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
    // Load the Datapath dialect so the rewriter can create Datapath ops
    ctx->loadDialect<DatapathDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<FoldAddReplicate>(ctx);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::Pass> circt::datapath::createDatapathReduceDelayPass() {
  return std::make_unique<DatapathReduceDelayPass>();
}