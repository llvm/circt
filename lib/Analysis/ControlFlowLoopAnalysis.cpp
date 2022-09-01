//===- ControlFlowLoopAnalysis.cpp - CF Loop Analysis ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions that perform loop analysis on structures
// expressed as a CFG.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/ControlFlowLoopAnalysis.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace circt::analysis;

namespace {
// The BFS callback provides the current block as an argument and returns
// whether search should halt.
enum class BFSNextState { Halt, SkipSuccessors, Continue, Custom };
using BFSCallbackExtended = llvm::function_ref<BFSNextState(
    Block *, DenseSet<Block *> &, SmallVector<Block *> &)>;

void blockBFS(Block *start, BFSCallbackExtended callback) {
  DenseSet<Block *> visited;
  SmallVector<Block *> queue = {start};
  while (!queue.empty()) {
    Block *currBlock = queue.front();
    queue.erase(queue.begin());
    if (visited.contains(currBlock))
      continue;
    visited.insert(currBlock);

    switch (callback(currBlock, visited, queue)) {
    case BFSNextState::Halt:
      return;
    case BFSNextState::Continue: {
      llvm::copy(currBlock->getSuccessors(), std::back_inserter(queue));
      break;
    }
    case BFSNextState::SkipSuccessors:
      for (auto *succ : currBlock->getSuccessors())
        visited.insert(succ);
      break;
    case BFSNextState::Custom:
      break;
    }
  }
}

using BFSCallback = llvm::function_ref<BFSNextState(Block *)>;
void blockBFS(Block *start, BFSCallback callback) {
  blockBFS(start, [&](Block *block, DenseSet<Block *> &,
                      SmallVector<Block *> &) { return callback(block); });
}

/// Performs a BFS to determine whether there exists a path between 'from' and
/// 'to'.
static bool isReachable(Block *from, Block *to) {
  bool isReachable = false;
  blockBFS(from, [&](Block *currBlock) {
    if (currBlock == to) {
      isReachable = true;
      return BFSNextState::Halt;
    }
    return BFSNextState::Continue;
  });
  return isReachable;
}

} // namespace

/// Helper that checks if entry is a loop header. If it is, it collects
/// additional information about the loop for further processing.
LogicalResult ControlFlowLoopAnalysis::collectLoopInfo(Block *entry,
                                                       LoopInfo &loopInfo) {
  loopInfo.loopHeader = entry;

  for (auto *backedge : entry->getPredecessors()) {
    bool dominates = domInfo.dominates(entry, backedge);
    bool formsLoop = isReachable(entry, backedge);
    if (formsLoop) {
      if (dominates)
        loopInfo.loopLatches.insert(backedge);
      else
        return entry->getParentOp()->emitError()
               << "Non-canonical loop structures detected; a potential "
                  "loop header has backedges not dominated by the loop "
                  "header. This indicates that the loop has multiple entry "
                  "points.";
    }
  }

  // Exit blocks are the blocks that control is transfered to after exiting
  // the loop. This is essentially determining the strongly connected
  // components with the loop header. We perform a BFS from the loop header,
  // and if the loop header is reachable from the block, it is within the
  // loop.
  blockBFS(entry, [&](Block *currBlock) {
    if (isReachable(currBlock, entry)) {
      loopInfo.inLoop.insert(currBlock);
      return BFSNextState::Continue;
    }
    loopInfo.exitBlocks.insert(currBlock);
    return BFSNextState::SkipSuccessors;
  });

  assert(loopInfo.inLoop.size() >= 2 && "A loop must have at least 2 blocks");
  assert(loopInfo.exitBlocks.size() != 0 &&
         "A loop must have an exit block...?");

  return success();
}

ControlFlowLoopAnalysis::ControlFlowLoopAnalysis(Region &region)
    : region(region), domInfo(region.getParentOp()) {}

bool ControlFlowLoopAnalysis::hasBackedge(Block *block) {
  return llvm::any_of(block->getPredecessors(),
                      [&](Block *pred) { return isReachable(block, pred); });
}

LogicalResult ControlFlowLoopAnalysis::analyzeRegion() {
  Block *entry = &region.front();
  LogicalResult result = success();
  blockBFS(entry, [&](Block *currBlock, DenseSet<Block *> &visited,
                      SmallVector<Block *> &queue) {
    if (!hasBackedge(currBlock))
      return BFSNextState::Continue;

    LoopInfo newInfo;
    if (failed(collectLoopInfo(currBlock, newInfo))) {
      result = failure();
      return BFSNextState::Halt;
    }

    // Adjusting the BFS state to jump over the loop.
    for (Block *loopBlock : newInfo.inLoop)
      visited.insert(loopBlock);
    llvm::copy(newInfo.exitBlocks, std::back_inserter(queue));

    topLevelLoops.emplace_back(std::move(newInfo));

    return BFSNextState::Custom;
  });

  return result;
}

bool ControlFlowLoopAnalysis::isLoopHeader(Block *b) {
  for (auto &info : topLevelLoops)
    if (info.loopHeader == b)
      return true;
  return false;
}

bool ControlFlowLoopAnalysis::isLoopElement(Block *b) {
  for (auto &info : topLevelLoops)
    if (info.inLoop.contains(b))
      return true;
  return false;
}

LoopInfo *ControlFlowLoopAnalysis::getLoopInfoForHeader(Block *b) {
  for (auto &info : topLevelLoops)
    if (info.loopHeader == b)
      return &info;

  return nullptr;
}

LoopInfo *ControlFlowLoopAnalysis::getLoopInfo(Block *b) {
  for (auto &info : topLevelLoops)
    if (info.inLoop.contains(b))
      return &info;

  return nullptr;
}
