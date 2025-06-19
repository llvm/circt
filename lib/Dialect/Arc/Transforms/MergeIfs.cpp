//===- MergeIfs.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-merge-ifs"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_MERGEIFSPASS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

namespace {
struct MergeIfsPass : public arc::impl::MergeIfsPassBase<MergeIfsPass> {
  void runOnOperation() override;
  void runOnBlock(Block &rootBlock);
  void sinkOps(Block &rootBlock);
  void mergeIfs(Block &rootBlock);

private:
  bool anyChanges;
};
} // namespace

void MergeIfsPass::runOnOperation() {
  // Go through the regions recursively, from outer regions to nested regions,
  // and try to move/sink/merge ops in each.
  getOperation()->walk<WalkOrder::PreOrder>([&](Region *region) {
    if (region->hasOneBlock() && mlir::mayHaveSSADominance(*region))
      runOnBlock(region->front());
  });
}

/// Iteratively sink ops into block, move them closer to their uses, and merge
/// adjacent `scf.if` operations.
void MergeIfsPass::runOnBlock(Block &rootBlock) {
  LLVM_DEBUG(llvm::dbgs() << "Running on block in "
                          << rootBlock.getParentOp()->getName() << "\n");
  do {
    ++numIterations;
    anyChanges = false;
    sinkOps(rootBlock);
    mergeIfs(rootBlock);
  } while (anyChanges);
}

/// Return the state/memory value being written by an op.
static Value getPointerWrittenByOp(Operation *op) {
  if (auto write = dyn_cast<StateWriteOp>(op))
    return write.getState();
  if (auto write = dyn_cast<MemoryWriteOp>(op))
    return write.getMemory();
  return {};
}

/// Return the state/memory value being read by an op.
static Value getPointerReadByOp(Operation *op) {
  if (auto read = dyn_cast<StateReadOp>(op))
    return read.getState();
  if (auto read = dyn_cast<MemoryReadOp>(op))
    return read.getMemory();
  return {};
}

/// Check if an operation has side effects, ignoring any nested ops. This is
/// useful if we're traversing all nested ops anyway, and we are only interested
/// in the current op's side effects.
static bool hasSideEffects(Operation *op) {
  if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(op))
    return !memEffects.hasNoEffect();
  return !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
}

namespace {
/// An integer indicating the position of an operation in its parent block. The
/// first field is the initial order/position assigned. The second field is used
/// to order ops that were moved to the same location, which makes them have the
/// same first field.
using OpOrder = std::pair<unsigned, unsigned>;

/// A helper that tracks an op and its order, and allows for convenient
/// substitution with another op that has a higher/lower order.
struct OpAndOrder {
  Operation *op = nullptr;
  OpOrder order = {0, 0};

  explicit operator bool() const { return op; }

  /// Assign `other` if its order is lower than this op, or this op is null.
  void minimize(const OpAndOrder &other) {
    if (!op || (other.op && other.order < order))
      *this = other;
  }

  /// Assign `other` if its order is higher than this op, or this op is null.
  void maximize(const OpAndOrder &other) {
    if (!op || (other.op && other.order > order))
      *this = other;
  }
};
} // namespace

/// Sink operations as close to their users as possible.
void MergeIfsPass::sinkOps(Block &rootBlock) {
  // A numeric position assigned to ops as we encounter them. Ops at the end of
  // the block get the lowest order number, ops at the beginning the highest.
  DenseMap<Operation *, OpOrder> opOrder;
  // A lookup table that indicates where ops should be inserted. This is used to
  // maintain the original op order if multiple ops pile up before the same
  // other op that blocks their move.
  DenseMap<Operation *, Operation *> insertionPoints;
  // The write ops to each state/memory pointer we've seen so far. ("Next"
  // because we run from the end to the beginning of the block.)
  DenseMap<Value, Operation *> nextWrite;
  // The most recent op that has an unknown (non-read/write) side-effect.
  Operation *nextSideEffect = nullptr;

  for (auto &op : llvm::make_early_inc_range(llvm::reverse(rootBlock))) {
    // Assign an order to this op.
    auto order = OpOrder{opOrder.size() + 1, 0};
    opOrder[&op] = order;
    // Track whether the op is, or contains, any writes (and thus can't
    // generally be moved into a block)
    bool opContainsWrites = false;

    // Analyze the side effects in the op.
    op.walk([&](Operation *subOp) {
      if (auto ptr = getPointerWrittenByOp(subOp)) {
        nextWrite[ptr] = &op;
        opContainsWrites = true;
      } else if (!isa<StateReadOp, MemoryReadOp>(subOp) &&
                 hasSideEffects(subOp)) {
        nextSideEffect = &op;
      }
    });

    // Determine how much the op can be moved.
    OpAndOrder moveLimit;
    if (auto ptr = getPointerReadByOp(&op)) {
      // Don't move across writes to the same state/memory.
      if (auto *write = nextWrite.lookup(ptr))
        moveLimit.maximize({write, opOrder.lookup(write)});
      // Don't move across general side-effecting ops.
      if (nextSideEffect)
        moveLimit.maximize({nextSideEffect, opOrder.lookup(nextSideEffect)});
    } else if (opContainsWrites || nextSideEffect == &op) {
      // Don't move writes or side-effecting ops.
      continue;
    }

    // Find the block that contains all uses.
    Block *allUsesInBlock = nullptr;
    for (auto *user : op.getUsers()) {
      // If this user is directly in the root block there's no chance of sinking
      // the current op anywhere.
      if (user->getBlock() == &rootBlock) {
        allUsesInBlock = nullptr;
        break;
      }

      // Find the operation in the root block that contains this user.
      while (user->getParentOp()->getBlock() != &rootBlock)
        user = user->getParentOp();
      assert(user);

      // Check that all users sit in the same op in the root block.
      if (!allUsesInBlock) {
        allUsesInBlock = user->getBlock();
      } else if (allUsesInBlock != user->getBlock()) {
        allUsesInBlock = nullptr;
        break;
      }
    }

    // If no single block exists that contains all uses, find the earliest op in
    // the root block that uses the current op.
    OpAndOrder earliest;
    if (allUsesInBlock) {
      earliest.op = allUsesInBlock->getParentOp();
      earliest.order = opOrder.lookup(earliest.op);
    } else {
      for (auto *user : op.getUsers()) {
        while (user->getBlock() != &rootBlock)
          user = user->getParentOp();
        assert(user);
        earliest.maximize({user, opOrder.lookup(user)});
      }
    }

    // Ensure we don't move past the move limit imposed by side effects.
    earliest.maximize(moveLimit);
    if (!earliest)
      continue;

    // Either move the op inside the single block that contains all uses, or
    // move it to just before its earliest user.
    if (allUsesInBlock && allUsesInBlock->getParentOp() == earliest.op) {
      op.moveBefore(allUsesInBlock, allUsesInBlock->begin());
      ++numOpsSunk;
      anyChanges = true;
      LLVM_DEBUG(llvm::dbgs() << "- Sunk " << op << "\n");
    } else {
      // Insert above other ops that we have already moved to this earliest op.
      // This ensures the original op order is maintained and we are not
      // spuriously flipping ops around. This also works without the
      // `insertionPoint` lookup, but can cause significant linear scanning to
      // find the op before which we want to insert.
      auto &insertionPoint = insertionPoints[earliest.op];
      if (insertionPoint) {
        auto order = opOrder.lookup(insertionPoint);
        assert(order.first == earliest.order.first);
        assert(order.second >= earliest.order.second);
        earliest.op = insertionPoint;
        earliest.order = order;
      }
      while (auto *prevOp = earliest.op->getPrevNode()) {
        auto order = opOrder.lookup(prevOp);
        if (order.first != earliest.order.first)
          break;
        assert(order.second > earliest.order.second);
        earliest.op = prevOp;
        earliest.order = order;
      }
      insertionPoint = earliest.op;

      // Only move if the op isn't already in the right spot.
      if (op.getNextNode() != earliest.op) {
        LLVM_DEBUG(llvm::dbgs() << "- Moved " << op << "\n");
        op.moveBefore(earliest.op);
        ++numOpsMovedToUser;
        anyChanges = true;
      }

      // Update the current op's order to reflect where it has been inserted.
      // This ensures that later moves to the same pile of moved ops do not
      // reorder the operations.
      order = earliest.order;
      assert(order.second < unsigned(-1));
      ++order.second;
      opOrder[&op] = order;
    }
  }
}

void MergeIfsPass::mergeIfs(Block &rootBlock) {
  DenseSet<Value> prevIfWrites, prevIfReads;

  scf::IfOp lastOp;
  for (auto ifOp : rootBlock.getOps<scf::IfOp>()) {
    auto prevIfOp = std::exchange(lastOp, ifOp);
    if (!prevIfOp)
      continue;

    // Only handle simple cases for now. (Same condition, no results, and both
    // ifs either have or don't have an else block.)
    if (ifOp.getCondition() != prevIfOp.getCondition())
      continue;
    if (ifOp.getNumResults() != 0 || prevIfOp.getNumResults() != 0)
      continue;
    if (ifOp.getElseRegion().empty() != prevIfOp.getElseRegion().empty())
      continue;

    // Try to move ops in between the `scf.if` ops above the previous `scf.if`
    // in order to make them immediately adjacent.
    if (ifOp->getPrevNode() != prevIfOp) {
      // Determine the side effects inside the previous if op.
      bool prevIfHasSideEffects = false;
      prevIfWrites.clear();
      prevIfReads.clear();
      prevIfOp.walk([&](Operation *op) {
        if (auto ptr = getPointerWrittenByOp(op))
          prevIfWrites.insert(ptr);
        else if (auto ptr = getPointerReadByOp(op))
          prevIfReads.insert(ptr);
        else if (!prevIfHasSideEffects && hasSideEffects(op))
          prevIfHasSideEffects = true;
      });

      // Check if it is legal to throw all ops over the previous `scf.if` op,
      // given the side effects. We don't move the ops yet to ensure we can move
      // *all* of them at once afterwards. Otherwise this optimization would
      // race with the sink-to-users optimization.
      bool allMovable = true;
      for (auto &op : llvm::make_range(Block::iterator(prevIfOp->getNextNode()),
                                       Block::iterator(ifOp))) {
        auto result = op.walk([&](Operation *subOp) {
          if (auto ptr = getPointerWrittenByOp(subOp)) {
            // We can't move writes over writes or reads of the same state.
            if (prevIfWrites.contains(ptr) || prevIfReads.contains(ptr))
              return WalkResult::interrupt();
          } else if (auto ptr = getPointerReadByOp(subOp)) {
            // We can't move reads over writes to the same state.
            if (prevIfWrites.contains(ptr))
              return WalkResult::interrupt();
          } else if (hasSideEffects(subOp)) {
            // We can't move side-effecting ops over other side-effecting ops.
            if (prevIfHasSideEffects)
              return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (result.wasInterrupted()) {
          allMovable = false;
          break;
        }
      }
      if (!allMovable)
        continue;

      // At this point we know that all ops can be moved. Do so.
      while (auto *op = prevIfOp->getNextNode()) {
        if (op == ifOp)
          break;
        LLVM_DEBUG(llvm::dbgs() << "- Moved before if " << *op << "\n");
        op->moveBefore(prevIfOp);
        ++numOpsMovedFromBetweenIfs;
      }
    }

    // Merge the then-blocks.
    prevIfOp.thenYield().erase();
    ifOp.thenBlock()->getOperations().splice(
        ifOp.thenBlock()->begin(), prevIfOp.thenBlock()->getOperations());

    // Merge the else-blocks if present.
    if (ifOp.elseBlock()) {
      prevIfOp.elseYield().erase();
      ifOp.elseBlock()->getOperations().splice(
          ifOp.elseBlock()->begin(), prevIfOp.elseBlock()->getOperations());
    }

    // Clean up.
    prevIfOp.erase();
    anyChanges = true;
    ++numIfsMerged;
    LLVM_DEBUG(llvm::dbgs() << "- Merged adjacent if ops\n");
  }
}
