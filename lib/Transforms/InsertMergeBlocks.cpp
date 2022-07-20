//===- InsertMergeBlocks.cpp - Insert Merge Blocks --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/InsertMergeBlocks.h"
#include "PassDetail.h"
#include "circt/Analysis/ControlFlowLoopAnalysis.h"
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
using namespace circt::analysis;

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

using PredMap = DenseMap<Block *, size_t>;
namespace {
/// A dual CFG that contracts cycles and irregular subgraphs into single logical
/// blocks.
struct DualGraph {
  DualGraph(Region &r, ControlFlowLoopAnalysis &loopAnalysis);

  size_t getNumPredecessors(Block *b) { return predCnts.lookup(b); }
  void getPredecessors(Block *b, SmallVectorImpl<Block *> &res);

  size_t getNumSuccessors(Block *b) { return succMap.lookup(b).size(); }
  ArrayRef<Block *> getSuccessors(Block *b) {
    return succMap.find(b)->getSecond();
  }

  Block *lookupHeaderBlock(Block *block);

  Region &r;
  ControlFlowLoopAnalysis &loopAnalysis;

  DenseMap<Block *, SmallVector<Block *>> succMap;
  PredMap predCnts;
};
} // namespace

DualGraph::DualGraph(Region &r, ControlFlowLoopAnalysis &loopAnalysis)
    : r(r), loopAnalysis(loopAnalysis), succMap(), predCnts() {
  for (Block &b : r) {
    if (!loopAnalysis.isLoopHeader(&b) && loopAnalysis.isLoopElement(&b))
      continue;

    SmallVector<Block *> succs;

    // NOTE: This assumes that there is only one exitting node, i.e., not
    // two blocks from the same loop can be predecessors of one block
    unsigned predCnt = 0;
    if (loopAnalysis.isLoopHeader(&b)) {
      LoopInfo *info = loopAnalysis.getLoopInfoForHeader(&b);
      for (auto *pred : b.getPredecessors())
        if (!info->inLoop.contains(pred))
          predCnt++;

      llvm::copy(info->exitBlocks, std::back_inserter(succs));
    } else {
      // TODO fix naming
      predCnt = ::getNumPredecessors(&b);
      llvm::copy(b.getSuccessors(), std::back_inserter(succs));
    }
    predCnts.try_emplace(&b, predCnt);
    succMap.try_emplace(&b, std::move(succs));
  }
}

Block *DualGraph::lookupHeaderBlock(Block *block) {
  // assumes that block is part of a loop
  LoopInfo *info = loopAnalysis.getLoopInfo(block);
  assert(info != nullptr);
  return info->loopHeader;
}

// TODO refactor
void DualGraph::getPredecessors(Block *b, SmallVectorImpl<Block *> &res) {
  assert((loopAnalysis.isLoopHeader(b) || !loopAnalysis.isLoopElement(b)) &&
         "can only get predecessors of blocks in the graph");

  if (loopAnalysis.isLoopHeader(b)) {
    LoopInfo *info = loopAnalysis.getLoopInfo(b);

    for (auto *pred : b->getPredecessors()) {
      if (info->inLoop.contains(pred))
        continue;

      if (loopAnalysis.isLoopElement(pred)) {
        // push back other loop header
        res.push_back(loopAnalysis.getLoopInfo(pred)->loopHeader);
        continue;
      }
      res.push_back(pred);
    }

    return;
  }

  for (auto *pred : b->getPredecessors()) {
    if (!loopAnalysis.isLoopElement(pred)) {
      res.push_back(pred);
      continue;
    }

    // NOTE: This will break down once multiple exit nodes are allowed
    res.push_back(loopAnalysis.getLoopInfo(pred)->loopHeader);
  }
}

static Block *getLastSplitBlock(SplitBlockMap &map, Block *block,
                                DualGraph &graph) {
  auto it = map.find(block);
  if (it == map.end()) {
    // if the block isn't part of the dual graph, we have to use the entry block
    // to the contracted region
    block = graph.lookupHeaderBlock(block);
    it = map.find(block);
  }
  assert(it != map.end() &&
         "expect block to have an entry in the split block map");
  SmallVector<Block *> &vec = it->getSecond();
  if (vec.empty())
    return nullptr;

  return vec.back();
}

/// Builds a binary merge block tree for the predecessors of currBlock.
static LogicalResult buildMergeBlocks(Block *currBlock,
                                      SplitBlockMap &prevSplitBlocks,
                                      Block *predDom,
                                      ConversionPatternRewriter &rewriter,
                                      DualGraph &graph) {
  llvm::SmallPtrSet<Block *, 4> preds;
  preds.insert(currBlock->getPredecessors().begin(),
               currBlock->getPredecessors().end());
  // Map from split blocks to blocks that descend from it.
  DenseMap<Block *, Block *> predsToConsider;

  while (!preds.empty()) {
    Block *pred;
    for (auto it = preds.begin(), end = preds.end(); it != end; ++it) {
      pred = *it;
      Block *splitBlock = getLastSplitBlock(prevSplitBlocks, pred, graph);
      if (splitBlock == predDom) {
        // Needs no additional merge block
        preds.erase(pred);
        continue;
      }

      if (predsToConsider.count(splitBlock) == 0) {
        // no other block with the same split block was found yet, so just store
        // it and continue
        predsToConsider.try_emplace(splitBlock, pred);
        preds.erase(pred);
        continue;
      }

      // Found a pair, so insert a new merge block for them
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

static LogicalResult preconditionCheck(Region &r,
                                       ControlFlowLoopAnalysis &analysis) {
  for (auto &info : analysis.topLevelLoops)
    if (info.exitBlocks.size() > 1)
      return r.getParentOp()->emitError(
          "multiple exit nodes are not yet supported");

  return success();
}

static void setupQueue(DualGraph &graph, SmallVector<Block *> &queue) {
  for (auto it : graph.predCnts)
    if (it.second == 0)
      queue.push_back(it.first);
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

  Block *entry = &r.front();
  DominanceInfo domInfo(r.getParentOp());
  PostDominanceInfo postDomInfo(r.getParentOp());

  ControlFlowLoopAnalysis loopAnalysis(r);
  if (failed(loopAnalysis.analyzeRegion()))
    return failure();

  if (failed(preconditionCheck(r, loopAnalysis)))
    return failure();

  // Traversing the graph in topological order
  SmallVector<Block *> queue;

  // Holds the graph that contains the relevant blocks. It for example contracts
  // loops into one block to preserve a DAG structure.
  DualGraph graph(r, loopAnalysis);
  setupQueue(graph, queue);

  // Counts the amount of predecessors remaining, if it reaches 0, insert into
  // queue.
  PredMap predsToVisit = graph.predCnts;

  SplitBlockMap prevSplitBlocks;
  prevSplitBlocks.try_emplace(entry, SmallVector<Block *>());

  while (!queue.empty()) {
    Block *currBlock = queue.front();
    queue.erase(queue.begin());

    auto it =
        prevSplitBlocks.try_emplace(currBlock, SmallVector<Block *>()).first;
    SmallVector<Block *> &currOut = it->getSecond();

    bool isMergeBlock = graph.getNumPredecessors(currBlock) > 1;
    bool isSplitBlock = graph.getNumSuccessors(currBlock) > 1;

    SmallVector<Block *> preds;
    graph.getPredecessors(currBlock, preds);

    if (isMergeBlock) {
      Block *predDom = currBlock;
      for (auto *pred : preds) {
        predDom = domInfo.findNearestCommonDominator(predDom, pred);
      }

      if (failed(buildMergeBlocks(currBlock, prevSplitBlocks, predDom, rewriter,
                                  graph)))
        return failure();

      // The predDom has similar properties as a normal predecessor, thus we can
      // just copy its split block information.
      SmallVector<Block *> predDomOut = prevSplitBlocks.lookup(predDom);
      llvm::copy(predDomOut, std::back_inserter(currOut));
    } else if (!preds.empty()) {
      Block *pred = preds.front();
      SmallVector<Block *> &predOut = prevSplitBlocks.find(pred)->getSecond();
      llvm::copy(predOut, std::back_inserter(currOut));
    }

    if (isSplitBlock) {
      currOut.push_back(currBlock);
    }

    for (auto *succ : graph.getSuccessors(currBlock)) {
      auto it = predsToVisit.find(succ);
      unsigned val = --(it->getSecond());
      // Pushing the block on the queue once all it's successors were visited
      // ensures a topological traversal.
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
