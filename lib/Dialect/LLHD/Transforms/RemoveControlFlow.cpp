//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-remove-control-flow"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_REMOVECONTROLFLOWPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {
/// A helper struct that tracks a boolean condition as either a constant false,
/// constant true, or an SSA value.
struct Condition {
  Condition() {}
  Condition(Value value) : pair(value, 0) {
    if (value) {
      if (matchPattern(value, m_One()))
        *this = Condition(true);
      if (matchPattern(value, m_Zero()))
        *this = Condition(false);
    }
  }
  Condition(bool konst) : pair(nullptr, konst ? 1 : 2) {}

  explicit operator bool() const {
    return pair.getPointer() != nullptr || pair.getInt() != 0;
  }

  bool isTrue() const { return !pair.getPointer() && pair.getInt() == 1; }
  bool isFalse() const { return !pair.getPointer() && pair.getInt() == 2; }
  Value getValue() const { return pair.getPointer(); }

  /// Turn this condition into an SSA value, creating an `hw.constant` if the
  /// condition is a constant.
  Value materialize(OpBuilder &builder, Location loc) const {
    if (isTrue())
      return hw::ConstantOp::create(builder, loc, APInt(1, 1));
    if (isFalse())
      return hw::ConstantOp::create(builder, loc, APInt(1, 0));
    return pair.getPointer();
  }

  Condition orWith(Condition other, OpBuilder &builder) const {
    if (isTrue() || other.isTrue())
      return true;
    if (isFalse())
      return other;
    if (other.isFalse())
      return *this;
    return builder.createOrFold<comb::OrOp>(getValue().getLoc(), getValue(),
                                            other.getValue());
  }

  Condition andWith(Condition other, OpBuilder &builder) const {
    if (isFalse() || other.isFalse())
      return false;
    if (isTrue())
      return other;
    if (other.isTrue())
      return *this;
    return builder.createOrFold<comb::AndOp>(getValue().getLoc(), getValue(),
                                             other.getValue());
  }

  Condition inverted(OpBuilder &builder) const {
    if (isTrue())
      return false;
    if (isFalse())
      return true;
    return comb::createOrFoldNot(getValue().getLoc(), getValue(), builder);
  }

private:
  llvm::PointerIntPair<Value, 2> pair;
};
} // namespace

/// Compute the branch decisions that cause control to flow from the dominator
/// to the target block.
///
/// TODO: This eagerly aggregates all control flow decisions. It may be more
/// efficient to first determine which blocks lie in between dominator and
/// target, and then only check that we are not taking decisions that cause
/// control flow to *leave* that set of blocks.
static Condition getBranchDecisionsFromDominatorToTarget(
    OpBuilder &builder, Block *dominator, Block *target,
    SmallDenseMap<std::pair<Block *, Block *>, Condition> &decisions) {
  if (auto decision = decisions.lookup({dominator, target}))
    return decision;

  SmallPtrSet<Block *, 8> visitedBlocks;
  visitedBlocks.insert(dominator); // stop at the dominator
  if (auto &decision = decisions[{dominator, dominator}]; !decision)
    decision = Condition(true);

  // Traverse the blocks in inverse post order. This ensures that we are
  // visiting all of a block's predecessors before we visit the block itself.
  // This allows us to first compute the decision leading control flow to each
  // of the predecessors, such that the current block can then just combine the
  // predecessor decisions.
  for (auto *block : llvm::inverse_post_order_ext(target, visitedBlocks)) {
    auto merged = Condition(false);
    for (auto *pred : block->getPredecessors()) {
      auto predDecision = decisions.lookup({dominator, pred});
      assert(predDecision);
      if (pred->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = cast<cf::CondBranchOp>(pred->getTerminator());
        if (condBr.getTrueDest() == condBr.getFalseDest()) {
          merged = merged.orWith(predDecision, builder);
        } else {
          auto cond = Condition(condBr.getCondition());
          if (condBr.getFalseDest() == block)
            cond = cond.inverted(builder);
          merged = merged.orWith(cond.andWith(predDecision, builder), builder);
        }
      } else {
        merged = merged.orWith(predDecision, builder);
      }
    }
    assert(merged);
    decisions.insert({{dominator, block}, merged});
  }

  return decisions.lookup({dominator, target});
}

//===----------------------------------------------------------------------===//
// Control Flow Removal
//===----------------------------------------------------------------------===//

namespace {
/// The main helper struct implementing control flow removal for a region.
struct CFRemover {
  CFRemover(Region &region) : region(region) {}
  void run();

  /// The region within which we are removing control flow.
  Region &region;
  /// The blocks in the region, sorted such that a block's predecessors appear
  /// in the list before the block itself.
  SmallVector<Block *> sortedBlocks;
  /// The dominance information for the region.
  DominanceInfo domInfo;
};
} // namespace

void CFRemover::run() {
  LLVM_DEBUG(llvm::dbgs() << "Removing control flow in " << region.getLoc()
                          << "\n");

  // Establish a topological order of the blocks in the region. Give up if we
  // detect a control flow cycle. Also take note of all YieldOps, such that we
  // can combine them into a single yield block later.
  SmallVector<YieldOp, 2> yieldOps;
  SmallPtrSet<Block *, 8> visitedBlocks, ipoSet;
  for (auto &block : region) {
    for (auto *ipoBlock : llvm::inverse_post_order_ext(&block, ipoSet)) {
      if (!llvm::all_of(ipoBlock->getPredecessors(), [&](auto *pred) {
            return visitedBlocks.contains(pred);
          })) {
        LLVM_DEBUG(llvm::dbgs() << "- Loop detected, giving up\n");
        return;
      }
      visitedBlocks.insert(ipoBlock);
      sortedBlocks.push_back(ipoBlock);
    }

    // Give up if there are any side-effecting ops in the region.
    for (auto &op : block) {
      if (!isMemoryEffectFree(&op)) {
        LLVM_DEBUG(llvm::dbgs() << "- Has side effects, giving up\n");
        return;
      }
    }

    // Check that we know what to do with all terminators.
    if (!isa<YieldOp, cf::BranchOp, cf::CondBranchOp>(block.getTerminator())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Has unsupported terminator "
                 << block.getTerminator()->getName() << ", giving up\n");
      return;
    }

    // Keep track of yield ops.
    if (auto yieldOp = dyn_cast<YieldOp>(block.getTerminator()))
      yieldOps.push_back(yieldOp);
  }

  // If there are multiple yield ops, factor them out into a single yield block.
  auto yieldOp = yieldOps[0];
  if (yieldOps.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "- Creating single yield block\n");
    OpBuilder builder(region.getContext());
    SmallVector<Location> locs(yieldOps[0].getNumOperands(), region.getLoc());
    auto *yieldBlock = builder.createBlock(&region, region.end(),
                                           yieldOps[0].getOperandTypes(), locs);
    sortedBlocks.push_back(yieldBlock);
    yieldOp =
        YieldOp::create(builder, region.getLoc(), yieldBlock->getArguments());
    for (auto yieldOp : yieldOps) {
      builder.setInsertionPoint(yieldOp);
      cf::BranchOp::create(builder, yieldOp.getLoc(), yieldBlock,
                           yieldOp.getOperands());
      yieldOp.erase();
    }
  }

  // Compute the dominance info for this region.
  domInfo = DominanceInfo(region.getParentOp());

  // Move operations into the entry block, replacing block arguments with
  // multiplexers as we go. The block order guarantees that we visit a block's
  // predecessors before we visit the block itself.
  SmallDenseMap<std::pair<Block *, Block *>, Condition> decisionCache;
  auto *entryBlock = sortedBlocks.front();
  for (auto *block : sortedBlocks) {
    if (!domInfo.isReachableFromEntry(block))
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "- Merging block ";
      block->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Find the nearest common dominator block of all predecessors. Any block
    // arguments reaching the current block will only depend on control flow
    // decisions between this dominator block and the current block.
    auto *domBlock = block;
    for (auto *pred : block->getPredecessors())
      if (domInfo.isReachableFromEntry(pred))
        domBlock = domInfo.findNearestCommonDominator(domBlock, pred);
    LLVM_DEBUG({
      llvm::dbgs() << "  - Common dominator: ";
      domBlock->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Convert the block arguments into multiplexers.
    OpBuilder builder(entryBlock->getTerminator());
    SmallVector<Value> mergedArgs;
    SmallPtrSet<Block *, 4> seenPreds;
    for (auto *pred : block->getPredecessors()) {
      // A block may be listed multiple times in the predecessors.
      if (!seenPreds.insert(pred).second)
        continue;

      // Only consider values coming from reachable predecessors.
      if (!domInfo.isReachableFromEntry(pred))
        continue;

      // Helper function to create a multiplexer between the current
      // `mergedArgs` and a new set of `args`, where the new args are picked if
      // `cond` is true and control flows from `domBlock` to `pred`. The
      // condition may be null, in which case the mux will directly use the
      // branch decisions that lead from `domBlock` to `pred`.
      auto mergeArgs = [&](ValueRange args, Condition cond, bool invCond) {
        if (mergedArgs.empty()) {
          mergedArgs = args;
          return;
        }
        auto decision = getBranchDecisionsFromDominatorToTarget(
            builder, domBlock, pred, decisionCache);
        if (cond) {
          if (invCond)
            cond = cond.inverted(builder);
          decision = decision.andWith(cond, builder);
        }
        for (auto [mergedArg, arg] : llvm::zip(mergedArgs, args)) {
          if (decision.isTrue())
            mergedArg = arg;
          else if (decision.isFalse())
            continue;
          else
            mergedArg = builder.createOrFold<comb::MuxOp>(
                arg.getLoc(), decision.materialize(builder, arg.getLoc()), arg,
                mergedArg);
        }
      };

      // Handle the different terminators that we support.
      if (auto condBrOp = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (condBrOp.getTrueDest() == condBrOp.getFalseDest()) {
          // Both destinations lead to the current block. Insert a mux to
          // collapse the destination operands and then treat this as an
          // unconditional branch to the current block.
          LLVM_DEBUG(llvm::dbgs() << "  - Both from " << condBrOp << "\n");
          SmallVector<Value> mergedOperands;
          mergedOperands.reserve(block->getNumArguments());
          for (auto [trueArg, falseArg] :
               llvm::zip(condBrOp.getTrueDestOperands(),
                         condBrOp.getFalseDestOperands())) {
            mergedOperands.push_back(builder.createOrFold<comb::MuxOp>(
                trueArg.getLoc(), condBrOp.getCondition(), trueArg, falseArg));
          }
          mergeArgs(mergedOperands, Value{}, false);
        } else if (condBrOp.getTrueDest() == block) {
          // The branch leads to the current block if the condition is true.
          LLVM_DEBUG(llvm::dbgs() << "  - True from " << condBrOp << "\n");
          mergeArgs(condBrOp.getTrueDestOperands(), condBrOp.getCondition(),
                    false);
        } else {
          // The branch leads to the current block if the condition is false.
          LLVM_DEBUG(llvm::dbgs() << "  - False from " << condBrOp << "\n");
          mergeArgs(condBrOp.getFalseDestOperands(), condBrOp.getCondition(),
                    true);
        }
      } else {
        auto brOp = cast<cf::BranchOp>(pred->getTerminator());
        LLVM_DEBUG(llvm::dbgs() << "  - From " << brOp << "\n");
        mergeArgs(brOp.getDestOperands(), Value{}, false);
      }
    }
    for (auto [blockArg, mergedArg] :
         llvm::zip(block->getArguments(), mergedArgs))
      blockArg.replaceAllUsesWith(mergedArg);

    // Move all ops except for the terminator into the entry block.
    if (block != entryBlock)
      entryBlock->getOperations().splice(--entryBlock->end(),
                                         block->getOperations(), block->begin(),
                                         --block->end());
  }

  // Move the yield op into the entry block, replacing the original terminator.
  if (yieldOp != entryBlock->getTerminator()) {
    yieldOp->moveBefore(entryBlock->getTerminator());
    entryBlock->getTerminator()->erase();
  }

  // Remove all blocks except for the entry block. We first clear all operations
  // in the blocks such that the blocks have no more uses in branch ops. Then we
  // remove the blocks themselves in a second pass.
  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->clear();
  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->erase();

  return;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct RemoveControlFlowPass
    : public llhd::impl::RemoveControlFlowPassBase<RemoveControlFlowPass> {
  void runOnOperation() override;
};
} // namespace

void RemoveControlFlowPass::runOnOperation() {
  for (auto op : getOperation().getOps<CombinationalOp>())
    CFRemover(op.getBody()).run();
}
