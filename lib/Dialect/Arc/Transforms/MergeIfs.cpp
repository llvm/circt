//===- MergeIfs.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Transforms/DialectConversion.h"
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
  using MergeIfsPassBase::MergeIfsPassBase;
  using MergeIfsPassBase::numIfsMerged;
  using MergeIfsPassBase::numOpsMovedFromBetweenIfs;
  using MergeIfsPassBase::numOpsMovedToUser;
  using MergeIfsPassBase::numOpsSunk;
};

using OpOrder = std::pair<unsigned, unsigned>;

struct OpAndOrder {
  Operation *op = nullptr;
  OpOrder order = {0, 0};

  explicit operator bool() const { return op; }

  void minimize(const OpAndOrder &other) {
    if (!op || (other.op && other.order < order))
      *this = other;
  }

  void maximize(const OpAndOrder &other) {
    if (!op || (other.op && other.order > order))
      *this = other;
  }
};

struct Sinker {
  MergeIfsPass &pass;
  Block &rootBlock;
  bool anyChanges;

  Sinker(MergeIfsPass &pass, Block &rootBlock)
      : pass(pass), rootBlock(rootBlock) {}
  LogicalResult run();
  void sinkOps();
  void mergeIfs();
};
} // namespace

void MergeIfsPass::runOnOperation() {
  auto result = getOperation()->walk<WalkOrder::PreOrder>([&](Region *region) {
    if (region->hasOneBlock() && mlir::mayHaveSSADominance(*region))
      if (failed(Sinker(*this, region->front()).run()))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    signalPassFailure();
}

LogicalResult Sinker::run() {
  LLVM_DEBUG(llvm::dbgs() << "Running on block in "
                          << rootBlock.getParentOp()->getName() << "\n");
  unsigned iteration = 0;
  do {
    if (iteration >= 10000)
      return rootBlock.getParentOp()->emitOpError()
             << "op sinking/merging did not converge";
    ++iteration;

    anyChanges = false;
    sinkOps();
    mergeIfs();
  } while (anyChanges);

  return success();
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

/// Sink operations as close to their users as possible.
void Sinker::sinkOps() {
  DenseMap<Operation *, OpOrder> opOrder;
  DenseMap<Operation *, Operation *> insertionPoints;
  DenseMap<Value, Operation *> nextWrite;
  Operation *nextSideEffect = nullptr;

  for (auto &op : llvm::make_early_inc_range(llvm::reverse(rootBlock))) {
    auto order = OpOrder{opOrder.size() + 1, 0};
    opOrder[&op] = order;

    // Analyze the side effects in the op.
    op.walk([&](Operation *subOp) {
      if (auto ptr = getPointerWrittenByOp(subOp))
        nextWrite[ptr] = &op;
      else if (!isa<StateReadOp, MemoryReadOp>(subOp) &&
               !subOp->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
               !mlir::isMemoryEffectFree(subOp))
        nextSideEffect = &op;
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
    } else if (isa<StateWriteOp, MemoryWriteOp>(&op) || nextSideEffect == &op) {
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

      // Zip up to the user's parent op immediately in the root block.
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
      ++pass.numOpsSunk;
      anyChanges = true;
      LLVM_DEBUG(llvm::dbgs() << "- Sunk " << op << "\n");
    } else {
      // LLVM_DEBUG(llvm::dbgs() << "- Moving " << op << " (order " <<
      // order.first
      //                         << "/" << order.second << ")\n");
      // LLVM_DEBUG(llvm::dbgs() << "  - Before " << *earliest.op << " (order "
      //                         << earliest.order.first << "/"
      //                         << earliest.order.second << ")\n");

      // Insert above other ops that we have already moved to this earliest op.
      // This ensures the original op order is maintained and we are not
      // spuriously flipping ops around.
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
        // LLVM_DEBUG(llvm::dbgs() << "  - Skipping " << *prevOp << " (order "
        //                         << order.first << "/" << order.second <<
        //                         ")\n");
        assert(order.second > earliest.order.second);
        earliest.op = prevOp;
        earliest.order = order;
      }
      insertionPoint = earliest.op;

      // auto &beforeOp = insertionPoints[earliest.op];
      // if (!beforeOp)
      //   beforeOp = earliest.op;
      // auto beforeOrder = opOrder.lookup(beforeOp);
      if (op.getNextNode() != earliest.op) {
        // LLVM_DEBUG(llvm::dbgs()
        //            << "  - Inserting before " << earliest.op->getName()
        //            << " (order " << earliest.order.first << "/"
        //            << earliest.order.second << ")\n");
        LLVM_DEBUG(llvm::dbgs() << "- Moved " << op << "\n");
        op.moveBefore(earliest.op);
        ++pass.numOpsMovedToUser;
        anyChanges = true;
      } else {
        // LLVM_DEBUG(llvm::dbgs() << "  - Already at correct location\n");
      }
      // beforeOp = &op;

      // Update the current op's order to reflect where it has been inserted.
      // This ensures that later moves to the same pile of moved ops do not
      // reorder the operations.
      order = earliest.order;
      assert(order.second < unsigned(-1));
      ++order.second;
      opOrder[&op] = order;
      // LLVM_DEBUG(llvm::dbgs() << "  - Updated order to " << order.first <<
      // "/"
      //                         << order.second << "\n");
    }
  }
}

void Sinker::mergeIfs() {
  DenseSet<Value> writes, reads;

  scf::IfOp lastOp;
  for (auto ifOp : rootBlock.getOps<scf::IfOp>()) {
    auto prevIfOp = std::exchange(lastOp, ifOp);
    if (!prevIfOp)
      continue;

    // Only handle simple cases for now.
    if (ifOp.getCondition() != prevIfOp.getCondition())
      continue;
    if (ifOp.getNumResults() != 0 || prevIfOp.getNumResults() != 0)
      continue;
    if (!ifOp.getThenRegion().hasOneBlock() ||
        !prevIfOp.getThenRegion().hasOneBlock())
      continue;
    if (!ifOp.getElseRegion().empty() &&
        (!ifOp.getElseRegion().hasOneBlock() ||
         !prevIfOp.getElseRegion().hasOneBlock()))
      continue;

    // Try to move ops in between the if ops above the previous if in order to
    // make them immediately adjacent.
    if (ifOp->getPrevNode() != prevIfOp) {
      // Determine the side effects inside the previous if op.
      bool hasSideEffects = false;
      writes.clear();
      reads.clear();
      prevIfOp.walk([&](Operation *op) {
        if (auto ptr = getPointerWrittenByOp(op))
          writes.insert(ptr);
        else if (auto ptr = getPointerReadByOp(op))
          reads.insert(ptr);
        else if (!hasSideEffects &&
                 !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
                 !mlir::isMemoryEffectFree(op))
          hasSideEffects = true;
      });

      // Check if it is legal to throw all ops over the previous if op, given
      // the side effects. We don't move the ops yet to ensure we can move *all*
      // of them at once afterwards. Otherwise this optimization would race with
      // the sink-to-users optimization.
      bool allMovable = true;
      for (auto &op : llvm::make_range(Block::iterator(prevIfOp->getNextNode()),
                                       Block::iterator(ifOp))) {
        auto result = op.walk([&](Operation *subOp) {
          if (auto ptr = getPointerWrittenByOp(subOp)) {
            // We can't move writes over writes or reads of the same state.
            if (writes.contains(ptr) || reads.contains(ptr))
              return WalkResult::interrupt();
          } else if (auto ptr = getPointerReadByOp(subOp)) {
            // We can't move reads over writes to the same state.
            if (writes.contains(ptr))
              return WalkResult::interrupt();
          } else if (!subOp->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
                     !mlir::isMemoryEffectFree(subOp)) {
            // We can't move side-effecting ops over other side-effecting ops.
            if (hasSideEffects)
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
        ++pass.numOpsMovedFromBetweenIfs;
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
    ++pass.numIfsMerged;
    LLVM_DEBUG(llvm::dbgs() << "- Merged adjacent if ops\n");
  }
}
