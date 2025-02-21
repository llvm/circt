//===- Balancing.cpp -AIG Binary Ops depth balancing ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers variadic AndInverter operations to binary AndInverter
// operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "aig-balancing"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_MAXIMUMANDCOVER
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

namespace {
struct MaximumAndCover : OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AndInverterOp op,
                                PatternRewriter &rewriter) const override {
    // multi-fanout node cannot be collapsed
    auto hasMultiFanout = [](AndInverterOp op) {
      int fanouts = 0;
      for (auto *user : op->getUsers()) {
        (void)user;
        fanouts++;
        if (fanouts > 1) {
          return true;
        }
      }
      return false;
    };

    llvm::SmallVector<Value> newFanins;
    llvm::SmallVector<bool> newInverts;

    for (auto [input, inverted] :
         llvm::zip(op.getOperands(), op.getInverted())) {
      auto andOp = input.getDefiningOp<aig::AndInverterOp>();
      if (inverted || !andOp || hasMultiFanout(op)) {
        newFanins.push_back(input);
        newInverts.push_back(inverted);
        continue;
      }

      // Collect fanin node's fanins into current node
      for (auto [fanin, faninInverted] :
           llvm::zip(andOp.getOperands(), andOp.getInverted())) {
        newFanins.push_back(fanin);
        newInverts.push_back(faninInverted);
      }
    }

    if (newFanins.size() != op.getOperands().size()) {
      rewriter.replaceOp(op, rewriter.createOrFold<aig::AndInverterOp>(
                                 op.getLoc(), newFanins, newInverts));
    }

    return failure();
  }
};

struct AIGMaximumAndCoverPass
    : public impl::MaximumAndCoverBase<AIGMaximumAndCoverPass> {
  void runOnOperation() override;
};
} // namespace

void AIGMaximumAndCoverPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<MaximumAndCover>(&getContext());

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  if (failed(mlir::applyPatternsGreedily(getOperation(), frozenPatterns)))
    return signalPassFailure();
}
