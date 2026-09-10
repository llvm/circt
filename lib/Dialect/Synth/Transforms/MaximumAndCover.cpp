//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs maximum AND-cover optimization by collapsing single-fanout
// AND nodes into their consumers.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "synth-maximum-and-cover"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_MAXIMUMANDCOVER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

namespace {
struct MaximumAndCoverPattern : OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(aig::AndInverterOp op,
                                PatternRewriter &rewriter) const override {
    // Check if any operand can be collapsed (single-fanout non-inverted AND)
    llvm::SmallVector<Value> newFanins;
    llvm::SmallVector<bool> newInverts;
    bool changed = false;

    for (auto [input, inverted] : llvm::zip(op.getInputs(), op.getInverted())) {
      auto andOp = input.getDefiningOp<aig::AndInverterOp>();

      // Can only collapse if:
      // 1. Input is an AND operation
      // 2. Input is not inverted in current op
      // 3. AND operation has only one use (single fanout)
      if (!inverted && andOp && andOp->hasOneUse()) {
        // Collect fanin node's inputs into current node
        for (auto [fanin, faninInverted] :
             llvm::zip(andOp.getInputs(), andOp.getInverted())) {
          newFanins.push_back(fanin);
          newInverts.push_back(faninInverted);
        }
        changed = true;
      } else {
        // Keep the original input
        newFanins.push_back(input);
        newInverts.push_back(inverted);
      }
    }

    if (!changed)
      return failure();

    // Create new AND operation with collapsed inputs
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, newFanins, newInverts);
    return success();
  }
};

struct MaximumAndCoverPass
    : public impl::MaximumAndCoverBase<MaximumAndCoverPass> {
  void runOnOperation() override;
};
} // namespace

void MaximumAndCoverPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<MaximumAndCoverPattern>(&getContext());

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(mlir::applyPatternsGreedily(getOperation(), frozenPatterns)))
    return signalPassFailure();
}
