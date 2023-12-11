//===- FuseWrites.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-fuse-writes"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_FUSEWRITES
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

namespace {
struct FuseWritesPass : public impl::FuseWritesBase<FuseWritesPass> {
  void runOnOperation() override;
};
} // namespace

void FuseWritesPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Fusing writes\n");

  using OpKey = std::tuple<Operation *, Block *, Value>;
  llvm::MapVector<OpKey, AllocStateOp> worklist, nextWorklist;
  getOperation().walk([&](StateWriteOp op) {
    if (auto stateOp = op.getState().getDefiningOp<AllocStateOp>())
      nextWorklist.insert({{op, op->getBlock(), op.getCondition()}, stateOp});
  });
  LLVM_DEBUG(llvm::dbgs() << "- " << nextWorklist.size() << " writes\n");
  llvm::EquivalenceClasses<AllocStateOp> relatedStates;
  for (unsigned depth = 0; depth < 2 && !nextWorklist.empty(); ++depth) {
    std::swap(worklist, nextWorklist);
    for (auto [opKey, state] : worklist) {
      auto [op, writeBlock, writeCondition] = opKey;
      for (auto operand : op->getOperands()) {
        if (isa<StateWriteOp>(op) && operand != op->getOperand(1))
          continue;
        if (isa<StateType, MemoryType, StorageType>(operand.getType()))
          continue;
        if (auto *defOp = operand.getDefiningOp()) {
          if (isa<AllocStateOp, RootInputOp, RootOutputOp, StateReadOp,
                  MemoryReadOp, MemoryWriteOp, MemoryReadPortOp,
                  MemoryWritePortOp>(defOp))
            continue;
          if (defOp->hasTrait<OpTrait::ConstantLike>())
            continue;
          auto it =
              nextWorklist.insert({{defOp, writeBlock, writeCondition}, state});
          relatedStates.unionSets(state, it.first->second);
        }
      }
    }
    worklist.clear();
  }
  LLVM_DEBUG(llvm::dbgs() << "- " << relatedStates.getNumClasses()
                          << " classes\n");

  std::vector<SmallVector<AllocStateOp>> statesToMerge;
  SmallDenseMap<Block *, SmallVector<AllocStateOp>> statesByBlock;
  for (auto classIt = relatedStates.begin(); classIt != relatedStates.end();
       ++classIt) {
    if (!classIt->isLeader())
      continue;

    for (auto stateIt = relatedStates.member_begin(classIt);
         stateIt != relatedStates.member_end(); ++stateIt)
      statesByBlock[(*stateIt)->getBlock()].push_back(*stateIt);

    for (auto &[block, states] : statesByBlock) {
      if (states.size() < 2)
        continue;
      llvm::sort(states, [](AllocStateOp a, AllocStateOp b) {
        auto sizeA = a.getType().getBitWidth();
        auto sizeB = b.getType().getBitWidth();
        if (sizeA > sizeB)
          return true;
        if (sizeA < sizeB)
          return false;
        return a->isBeforeInBlock(b);
      });
      // LLVM_DEBUG({
      //   llvm::dbgs() << "- Class:\n";
      //   for (auto state : states)
      //     llvm::dbgs() << "  - " << state << "\n";
      // });
      statesToMerge.push_back(std::move(states));
    }

    statesByBlock.clear();
  }
  LLVM_DEBUG(llvm::dbgs() << "- " << statesToMerge.size()
                          << " groups to merge\n");

  for (auto &states : statesToMerge) {
    SmallVector<hw::StructType::FieldInfo> elements;
    for (auto [index, state] : llvm::enumerate(states))
      elements.push_back({StringAttr::get(&getContext(), "f" + Twine(index)),
                          state.getType().getType()});
    auto type = hw::StructType::get(&getContext(), elements);
    LLVM_DEBUG(llvm::dbgs() << "  - Merging " << states.size()
                            << " states into " << type << "\n");

    OpBuilder builder(states[0]);
    auto newStateOp = builder.create<AllocStateOp>(
        states[0].getLoc(), StateType::get(type), states[0].getStorage());

    // Create the compound write.
    SmallVector<Operation *> writeOps;
    SmallVector<Value> writeValues;
    StateWriteOp lastWriteOp;
    bool multiplyWritten = false;
    for (auto state : states) {
      StateWriteOp singleWriteOp;
      for (auto *user : state->getUsers()) {
        auto writeOp = dyn_cast<StateWriteOp>(user);
        if (!writeOp)
          continue;
        if (singleWriteOp) {
          multiplyWritten = true;
          break;
        }
        singleWriteOp = writeOp;
      }
      if (multiplyWritten)
        break;
      assert(!lastWriteOp ||
             lastWriteOp.getCondition() == singleWriteOp.getCondition());
      if (!lastWriteOp || lastWriteOp->isBeforeInBlock(singleWriteOp))
        lastWriteOp = singleWriteOp;
      writeOps.push_back(singleWriteOp);
      writeValues.push_back(singleWriteOp.getValue());
    }
    if (multiplyWritten)
      continue;
    assert(writeValues.size() == elements.size());
    assert(lastWriteOp);

    builder.setInsertionPoint(lastWriteOp);
    auto aggregateValue = builder.create<hw::StructCreateOp>(
        lastWriteOp.getLoc(), type, writeValues);
    builder.create<StateWriteOp>(lastWriteOp.getLoc(), newStateOp,
                                 aggregateValue, lastWriteOp.getCondition());

    for (auto *op : writeOps)
      op->erase();

    // Replace all reads.
    SmallDenseMap<Block *, SmallVector<StateReadOp>> readsByBlock;
    for (auto state : states)
      for (auto *user : state->getUsers())
        readsByBlock[user->getBlock()].push_back(cast<StateReadOp>(user));
    SmallDenseMap<Block *, StateReadOp> newReadByBlock;
    for (auto &[block, reads] : readsByBlock) {
      llvm::sort(reads, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      builder.setInsertionPoint(reads[0]);
      newReadByBlock[block] =
          builder.create<StateReadOp>(reads[0].getLoc(), newStateOp);
    }

    for (auto [index, state] : llvm::enumerate(states)) {
      for (auto *user : llvm::make_early_inc_range(state->getUsers())) {
        auto readOp = dyn_cast<StateReadOp>(user);
        auto newReadOp = newReadByBlock.lookup(readOp->getBlock());
        builder.setInsertionPoint(readOp);
        auto field = builder.create<hw::StructExtractOp>(
            readOp.getLoc(), newReadOp, elements[index].name);
        readOp.replaceAllUsesWith(field.getResult());
        readOp.erase();
      }
      state.erase();
    }
  }
}
