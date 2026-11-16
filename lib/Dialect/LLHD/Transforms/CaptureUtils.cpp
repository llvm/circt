//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CaptureUtils.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#define DEBUG_TYPE "llhd-capture-utils"

using namespace mlir;
using namespace circt;
using namespace llhd;

/// Add a probe as block argument to a list of wait ops and update uses of the
/// probe to use the added block arguments as appropriate. This may insert
/// additional block arguments in case the probe and added block arguments both
/// reach the same block.
void llhd::captureAcrossWait(Value value, ArrayRef<WaitOp> waitOps,
                             Liveness &liveness, DominanceInfo &dominance) {
  // Constant-like values are guaranteed not to change across a wait and can
  // be rematerialized freely, so they never need to be captured. Everything
  // else defined inside the process, including time and ref values derived
  // from mutable state, must be captured so that uses after the wait observe
  // the value from before the wait. Values defined outside the process are
  // filtered by the caller.
  if (auto *defOp = value.getDefiningOp())
    if (defOp->hasTrait<OpTrait::ConstantLike>())
      return;

  SmallVector<WaitOp> capturedWaitOps;
  for (auto waitOp : waitOps) {
    auto *waitBlock = waitOp->getBlock();
    auto *blockLiveness = liveness.getLiveness(waitBlock);
    if (!blockLiveness || !blockLiveness->isLiveOut(value))
      continue;
    if (!dominance.dominates(value, waitOp.getOperation()))
      continue;
    capturedWaitOps.push_back(waitOp);
  }
  if (capturedWaitOps.empty())
    return;

  LLVM_DEBUG({
    llvm::dbgs() << "Capture " << value << "\n";
    for (auto waitOp : capturedWaitOps)
      llvm::dbgs() << "- Across " << waitOp << "\n";
  });

  // Calculate the merge points for this probe once it gets promoted to block
  // arguments across the wait ops.
  Region &region = *value.getParentRegion();
  auto &domTree = dominance.getDomTree(&region);
  llvm::IDFCalculatorBase<Block, false> idfCalculator(domTree);

  // Calculate the set of blocks which will define this probe as a distinct
  // value.
  SmallPtrSet<Block *, 4> definingBlocks;
  definingBlocks.insert(value.getParentBlock());
  for (auto waitOp : capturedWaitOps)
    definingBlocks.insert(waitOp.getDest());
  idfCalculator.setDefiningBlocks(definingBlocks);

  // Calculate where the probe is live.
  SmallPtrSet<Block *, 16> liveInBlocks;
  for (auto &block : region)
    if (liveness.getLiveness(&block)->isLiveIn(value))
      liveInBlocks.insert(&block);
  idfCalculator.setLiveInBlocks(liveInBlocks);

  // Calculate the merge points where we will have to insert block arguments for
  // this probe.
  SmallVector<Block *> mergePointsVec;
  idfCalculator.calculate(mergePointsVec);
  SmallPtrSet<Block *, 16> mergePoints(mergePointsVec.begin(),
                                       mergePointsVec.end());
  for (auto waitOp : capturedWaitOps)
    mergePoints.insert(waitOp.getDest());
  LLVM_DEBUG(llvm::dbgs() << "- " << mergePoints.size() << " merge points\n");

  // Perform a depth-first search starting at the block containing the probe,
  // which dominates all its uses. When we encounter a block that is a merge
  // point, insert a block argument.
  struct WorklistItem {
    DominanceInfoNode *domNode;
    Value reachingDef;
  };
  SmallVector<WorklistItem> worklist;
  worklist.push_back({domTree.getNode(value.getParentBlock()), value});

  while (!worklist.empty()) {
    auto item = worklist.pop_back_val();
    auto *block = item.domNode->getBlock();

    // If this block is a merge point, insert a block argument for the probe.
    if (mergePoints.contains(block))
      item.reachingDef = block->addArgument(value.getType(), value.getLoc());

    // Replace any uses of the probe in this block with the current reaching
    // definition.
    for (auto &op : *block)
      op.replaceUsesOfWith(value, item.reachingDef);

    // If the terminator of this block branches to a merge point, add the
    // current reaching definition as a destination operand.
    if (auto branchOp = dyn_cast<BranchOpInterface>(block->getTerminator())) {
      for (auto &blockOperand : branchOp->getBlockOperands())
        if (mergePoints.contains(blockOperand.get()))
          branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
              .append(item.reachingDef);
    } else if (auto waitOp = dyn_cast<WaitOp>(block->getTerminator())) {
      if (mergePoints.contains(waitOp.getDest()))
        waitOp.getDestOperandsMutable().append(item.reachingDef);
    }

    for (auto *child : item.domNode->children())
      worklist.push_back({child, item.reachingDef});
  }
}
