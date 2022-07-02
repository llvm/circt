//===- InsertMergeBlocks.cpp - Insert Merge Blocks --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/InsertMergeBlocks.h"
#include "PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

static size_t getNumPredecessors(Block *block) {
  return std::distance(block->getPredecessors().begin(),
                       block->getPredecessors().end());
}

/// Replaces the branching to oldDest of with an equivalent operation that
/// instead branches to newDest
static LogicalResult changeBranchTarget(Block *block, Block *oldDest,
                                        Block *newDest,
                                        ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointToEnd(block);
  auto term = block->getTerminator();
  return llvm::TypeSwitch<Operation *, LogicalResult>(term)
      .Case<cf::BranchOp>([&](auto branchOp) {
        rewriter.replaceOpWithNewOp<cf::BranchOp>(branchOp, newDest,
                                                  branchOp->getOperands());
        return success();
      })
      .Case<cf::CondBranchOp>([&](auto condBr) {
        auto cond = condBr.getCondition();
        ValueRange trueOperands = condBr.getTrueOperands();
        ValueRange falseOperands = condBr.getFalseOperands();

        Block *trueDest = condBr.getTrueDest();
        Block *falseDest = condBr.getFalseDest();

        // change to the correct destination
        if (trueDest == oldDest)
          trueDest = newDest;

        if (falseDest == oldDest)
          falseDest = newDest;

        rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
            condBr, cond, trueDest, trueOperands, falseDest, falseOperands);
        return success();
      })
      .Default([&](Operation *op) {
        return op->emitError("Unexpected terminator that cannot be handled.");
      });
}

/// Creates a new intermediate block that b1 and b2 branch to. The new block
/// branches to their common successor oldSucc.
static Block *buildMergeBlock(Block *b1, Block *b2, Block *oldSucc,
                              ConversionPatternRewriter &rewriter) {
  auto blockArgTypes = oldSucc->getArgumentTypes();

  Block *res =
      rewriter.createBlock(oldSucc, blockArgTypes, rewriter.getUnknownLoc());
  rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(), oldSucc,
                                res->getArguments());

  if (failed(changeBranchTarget(b1, oldSucc, res, rewriter)))
    return nullptr;
  if (failed(changeBranchTarget(b2, oldSucc, res, rewriter)))
    return nullptr;

  return res;
}

using SplitBlockMap = DenseMap<Block *, SmallVector<Block *>>;

static Block *getLastSplitBlock(SplitBlockMap &map, Block *block) {
  SmallVector<Block *> vec = map.lookup(block);
  if (vec.empty())
    return nullptr;

  return vec.back();
}

/// Builds a binary merge block tree for the predecessors of currBlock.
static LogicalResult buildMergeBlocks(Block *currBlock,
                                      SplitBlockMap &prevSplitBlocks,
                                      Block *predDom,
                                      ConversionPatternRewriter &rewriter) {
  llvm::SmallPtrSet<Block *, 4> preds;
  preds.insert(currBlock->getPredecessors().begin(),
               currBlock->getPredecessors().end());
  // Map from split blocks to blocks that descend from it.
  DenseMap<Block *, Block *> predsToConsider;

  while (!preds.empty()) {
    Block *pred;
    for (auto it = preds.begin(), end = preds.end(); it != end;
         pred = *(it++)) {
      Block *splitBlock = getLastSplitBlock(prevSplitBlocks, pred);
      if (splitBlock == predDom) {
        // Needs no additional merge block
        preds.erase(pred);
        continue;
      }

      if (predsToConsider.count(splitBlock) == 0) {
        predsToConsider.try_emplace(splitBlock, pred);
        preds.erase(pred);
        continue;
      }

      Block *other = predsToConsider.lookup(splitBlock);
      predsToConsider.erase(splitBlock);
      preds.erase(pred);

      Block *mergeBlock = buildMergeBlock(pred, other, currBlock, rewriter);
      if (!mergeBlock)
        return failure();

      // update info for the newly created block
      auto prevSplitIt =
          prevSplitBlocks.try_emplace(mergeBlock, SmallVector<Block *>()).first;
      SmallVector<Block *> &mergeBlockOut = prevSplitIt->getSecond();

      llvm::copy(llvm::drop_end(prevSplitBlocks.lookup(pred)),
                 std::back_inserter(mergeBlockOut));

      it = preds.insert(mergeBlock).first;
    }
  }
  return success();
}

/// Insert additional blocks that serve as counterparts to the blocks that
/// diverged the control flow.
/// The resulting merge block tree is guaranteed to be a binary tree.
///
/// This transformation does not affect any blocks that are part of a loop.
LogicalResult
circt::insertExplicitMergeBlocks(Region &r,
                                 ConversionPatternRewriter &rewriter) {
  // TODO relax these assumptions
  // Assumptions for now
  // - No unstructured control flow
  // - No loops

  // List approach

  Block *entry = &r.front();
  DominanceInfo domInfo(r.getParentOp());
  PostDominanceInfo postDomInfo(r.getParentOp());

  // Traversing the graph in topological order
  SmallVector<Block *> queue;

  // Counts the amount of predecessors remaining, if it reaches 0, insert into
  // queue.
  DenseMap<Block *, unsigned> predsToVisit;
  for (Block &b : r) {
    // TODO changes for loops
    unsigned predCnt = getNumPredecessors(&b);
    predsToVisit.try_emplace(&b, predCnt);
    if (predCnt == 0)
      queue.push_back(&b);
  }

  SplitBlockMap prevSplitBlocks;
  prevSplitBlocks.try_emplace(entry, SmallVector<Block *>());

  while (!queue.empty()) {
    Block *currBlock = queue.front();
    queue.erase(queue.begin());

    auto it =
        prevSplitBlocks.try_emplace(currBlock, SmallVector<Block *>()).first;
    SmallVector<Block *> &currOut = it->getSecond();

    bool isMergeBlock = getNumPredecessors(currBlock) > 1;
    bool isSplitBlock = currBlock->getNumSuccessors() > 1;

    if (isMergeBlock) {
      Block *predDom = currBlock;
      for (auto *pred : currBlock->getPredecessors()) {
        predDom = domInfo.findNearestCommonDominator(predDom, pred);
      }

      if (failed(
              buildMergeBlocks(currBlock, prevSplitBlocks, predDom, rewriter)))
        return failure();

      // Have to search manually through all predecessors, as the old ones are
      // still listed. This is due to the rewriter not materializing the
      // changes.
      SmallPtrSet<Block *, 2> remainingPreds;
      for (Block *pred : currBlock->getPredecessors()) {
        if (getLastSplitBlock(prevSplitBlocks, pred) == predDom)
          remainingPreds.insert(pred);
      }

      assert(remainingPreds.size() == 2);

      auto predIt = remainingPreds.begin();
      Block *pred0 = *predIt;
      Block *pred1 = *(++predIt);

      SmallVector<Block *> pred0Out = prevSplitBlocks.lookup(pred0);
      SmallVector<Block *> pred1Out = prevSplitBlocks.lookup(pred1);
      assert(pred0Out == pred1Out);

      llvm::copy(llvm::drop_end(pred0Out), std::back_inserter(currOut));
    } else {
      Block *pred = currBlock->getSinglePredecessor();
      SmallVector<Block *> predOut = prevSplitBlocks.lookup(pred);
      llvm::copy(predOut, std::back_inserter(currOut));
    }

    if (isSplitBlock) {
      currOut.push_back(currBlock);
    }

    for (auto *succ : currBlock->getSuccessors()) {
      auto it = predsToVisit.find(succ);
      unsigned val = --(it->getSecond());
      if (val == 0)
        queue.push_back(succ);
    }
  }

  return success();
}

namespace {

using PtrSet = SmallPtrSet<Operation *, 4>;

struct FuncOpPattern : public OpConversionPattern<func::FuncOp> {

  FuncOpPattern(PtrSet &rewrittenFuncs, MLIRContext *ctx)
      : OpConversionPattern(ctx), rewrittenFuncs(rewrittenFuncs) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);

    if (!op.isExternal())
      if (failed(insertExplicitMergeBlocks(op.getRegion(), rewriter)))
        return failure();

    rewriter.finalizeRootUpdate(op);

    // Insert this function into the set of processed functions
    rewrittenFuncs.insert(op);

    return success();
  }

private:
  PtrSet &rewrittenFuncs;
};

struct InsertMergeBlocksPass
    : public InsertMergeBlocksBase<InsertMergeBlocksPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    // Remembers traversed functions to only apply the conversion once
    PtrSet rewrittenFuncs;
    patterns.add<FuncOpPattern>(rewrittenFuncs, ctx);

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp func) { return rewrittenFuncs.contains(func); });
    target.addLegalDialect<cf::ControlFlowDialect>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createInsertMergeBlocksPass() {
  return std::make_unique<InsertMergeBlocksPass>();
}
} // namespace circt
