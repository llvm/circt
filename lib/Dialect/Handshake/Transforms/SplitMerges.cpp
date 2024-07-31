//===- SplitMerges.cpp - handshake merge deconstruction pass --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the handshake merge deconstruction pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace handshake {
#define GEN_PASS_DEF_HANDSHAKESPLITMERGES
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"
} // namespace handshake
} // namespace circt

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace {

struct DeconstructMergePattern : public OpRewritePattern<handshake::MergeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MergeOp mergeOp,
                                PatternRewriter &rewriter) const override {
    if (mergeOp.getNumOperands() <= 2)
      return failure();

    llvm::SmallVector<Value> mergeInputs;
    llvm::copy(mergeOp.getOperands(), std::back_inserter(mergeInputs));

    // Recursively build a balanced 2-input merge tree.
    while (mergeInputs.size() > 1) {
      llvm::SmallVector<Value> newMergeInputs;
      for (unsigned i = 0, e = mergeInputs.size(); i < ((e / 2) * 2); i += 2) {
        auto cm2 = rewriter.create<handshake::MergeOp>(
            mergeOp.getLoc(), ValueRange{mergeInputs[i], mergeInputs[i + 1]});
        newMergeInputs.push_back(cm2.getResult());
      }
      if (mergeInputs.size() % 2 != 0)
        newMergeInputs.push_back(mergeInputs.back());

      mergeInputs = newMergeInputs;
    }

    assert(mergeInputs.size() == 1);
    rewriter.replaceOp(mergeOp, mergeInputs[0]);

    return success();
  }
};

struct DeconstructCMergePattern
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    if (cmergeOp.getNumOperands() <= 2)
      return failure();

    Type cmergeIndexType = cmergeOp.getIndex().getType();
    auto loc = cmergeOp.getLoc();

    // Function for create a cmerge-pack structure which generates a
    // tuple<index, data> from two operands and an index offset.
    auto mergeTwoOperands = [&](Value op0, Value op1,
                                unsigned idxOffset) -> Value {
      auto cm2 = rewriter.create<handshake::ControlMergeOp>(
          loc, ValueRange{op0, op1}, cmergeIndexType);
      Value idxOperand = cm2.getIndex();
      if (idxOffset != 0) {
        // Non-zero index offset; add it to the index operand.
        idxOperand = rewriter.create<arith::AddIOp>(
            loc, idxOperand,
            rewriter.create<arith::ConstantOp>(
                loc, rewriter.getIntegerAttr(cmergeIndexType, idxOffset)));
      }

      // Pack index and data into a tuple s.t. they share control.
      return rewriter.create<handshake::PackOp>(
          loc, ValueRange{cm2.getResult(), idxOperand});
    };

    llvm::SmallVector<Value> packedTuples;
    // Perform the two-operand merges.
    for (unsigned i = 0, e = cmergeOp.getNumOperands(); i < ((e / 2) * 2);
         i += 2) {
      packedTuples.push_back(mergeTwoOperands(cmergeOp.getOperand(i),
                                              cmergeOp.getOperand(i + 1), i));
    }
    if (cmergeOp.getNumOperands() % 2 != 0) {
      // If there is an odd number of operands, the last operand becomes a tuple
      // of itself with an index of the number of operands - 1.
      unsigned lastIdx = cmergeOp.getNumOperands() - 1;
      packedTuples.push_back(rewriter.create<handshake::PackOp>(
          loc, ValueRange{cmergeOp.getOperand(lastIdx),
                          rewriter.create<arith::ConstantOp>(
                              loc, rewriter.getIntegerAttr(cmergeIndexType,
                                                           lastIdx))}));
    }

    // Non-deterministically merge the tuples and unpack the result.
    auto mergedTuple =
        rewriter.create<handshake::MergeOp>(loc, ValueRange(packedTuples));

    // And finally, replace the original cmerge with the unpacked result.
    rewriter.replaceOpWithNewOp<handshake::UnpackOp>(cmergeOp,
                                                     mergedTuple.getResult());
    return success();
  }
};

struct HandshakeSplitMerges
    : public circt::handshake::impl::HandshakeSplitMergesBase<
          HandshakeSplitMerges> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<DeconstructCMergePattern, DeconstructMergePattern>(
        &getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::Pass> circt::handshake::createHandshakeSplitMergesPass() {
  return std::make_unique<HandshakeSplitMerges>();
}
