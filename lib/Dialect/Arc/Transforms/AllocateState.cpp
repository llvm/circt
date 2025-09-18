//===- AllocateState.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-allocate-state"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_ALLOCATESTATE
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

using llvm::SmallMapVector;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

struct AllocateStatePass
    : public arc::impl::AllocateStateBase<AllocateStatePass> {

  using AllocateStateBase::AllocateStateBase;

  void runOnOperation() override;
  void allocateBlock(Block *block);
  void allocateOps(Value storage, Block *block, ArrayRef<Operation *> ops);

  void tapNamedState(AllocStateOp allocOp, mlir::Type valueType,
                     IntegerAttr offset);

  SmallVector<Attribute> traceTaps;
};
} // namespace

void AllocateStatePass::runOnOperation() {
  traceTaps.clear();

  ModelOp modelOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Allocating state in `" << modelOp.getName()
                          << "`\n");

  // Walk the blocks from innermost to outermost and group all state allocations
  // in that block in one larger allocation.
  modelOp.walk([&](Block *block) { allocateBlock(block); });

  if (!traceTaps.empty())
    modelOp.setTraceTapsAttr(ArrayAttr::get(modelOp.getContext(), traceTaps));
}

void AllocateStatePass::tapNamedState(AllocStateOp allocOp,
                                      mlir::Type valueType,
                                      IntegerAttr offset) {
  TraceTapAttr tapAttr;
  if (auto namesAttr = allocOp->getAttrOfType<ArrayAttr>("names")) {
    if (!namesAttr.empty()) {
      auto typeAttr = TypeAttr::get(valueType);
      tapAttr = TraceTapAttr::get(allocOp.getContext(), typeAttr,
                                  offset.getValue().getZExtValue(), namesAttr);
    }
  } else if (auto nameAttr = allocOp->getAttrOfType<StringAttr>("name")) {
    auto typeAttr = TypeAttr::get(valueType);
    auto arrayAttr = ArrayAttr::get(nameAttr.getContext(), {nameAttr});
    tapAttr = TraceTapAttr::get(allocOp.getContext(), typeAttr,
                                offset.getValue().getZExtValue(), arrayAttr);
  }
  if (!tapAttr)
    return;

  auto indexAttr =
      IntegerAttr::get(IndexType::get(allocOp.getContext()), traceTaps.size());
  traceTaps.push_back(tapAttr);
  for (auto user : allocOp.getResult().getUsers()) {
    if (auto writeUser = dyn_cast<StateWriteOp>(user)) {
      writeUser.setTraceTapIndexAttr(indexAttr);
      writeUser.setTraceTapModelAttr(getOperation().getNameAttr());
    }
  }
}

void AllocateStatePass::allocateBlock(Block *block) {
  SmallMapVector<Value, std::vector<Operation *>, 1> opsByStorage;

  // Group operations by their storage. There is generally just one storage,
  // passed into the model as a block argument.
  for (auto &op : *block) {
    if (isa<AllocStateOp, RootInputOp, RootOutputOp, AllocMemoryOp,
            AllocStorageOp>(&op))
      opsByStorage[op.getOperand(0)].push_back(&op);
  }
  LLVM_DEBUG(llvm::dbgs() << "- Visiting block in "
                          << block->getParentOp()->getName() << "\n");

  // Actually allocate each operation.
  for (auto &[storage, ops] : opsByStorage)
    allocateOps(storage, block, ops);
}

void AllocateStatePass::allocateOps(Value storage, Block *block,
                                    ArrayRef<Operation *> ops) {
  SmallVector<std::tuple<Value, Value, IntegerAttr>> gettersToCreate;

  // Helper function to allocate storage aligned to its own size, or 8 bytes at
  // most.
  unsigned currentByte = 0;
  auto allocBytes = [&](unsigned numBytes) {
    currentByte = llvm::alignToPowerOf2(
        currentByte, llvm::bit_ceil(std::min(numBytes, 16U)));
    unsigned offset = currentByte;
    currentByte += numBytes;
    return offset;
  };

  // Allocate storage for the operations.
  OpBuilder builder(block->getParentOp());
  for (auto *op : ops) {
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      auto result = op->getResult(0);
      auto storage = op->getOperand(0);
      unsigned numBytes = cast<StateType>(result.getType()).getByteWidth();
      auto offset = builder.getI32IntegerAttr(allocBytes(numBytes));
      op->setAttr("offset", offset);
      gettersToCreate.emplace_back(result, storage, offset);
      if (insertTraceTaps)
        if (auto allocStateOp = dyn_cast<AllocStateOp>(op))
          tapNamedState(allocStateOp,
                        cast<StateType>(result.getType()).getType(), offset);
      continue;
    }

    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto memType = memOp.getType();
      unsigned stride = memType.getStride();
      unsigned numBytes = memType.getNumWords() * stride;
      auto offset = builder.getI32IntegerAttr(allocBytes(numBytes));
      op->setAttr("offset", offset);
      op->setAttr("stride", builder.getI32IntegerAttr(stride));
      gettersToCreate.emplace_back(memOp, memOp.getStorage(), offset);
      continue;
    }

    if (auto allocStorageOp = dyn_cast<AllocStorageOp>(op)) {
      auto offset = builder.getI32IntegerAttr(
          allocBytes(allocStorageOp.getType().getSize()));
      allocStorageOp.setOffsetAttr(offset);
      gettersToCreate.emplace_back(allocStorageOp, allocStorageOp.getInput(),
                                   offset);
      continue;
    }

    assert("unsupported op for allocation" && false);
  }

  // For every user of the alloc op, create a local `StorageGetOp`.
  // First, create an ordering of operations to avoid a very expensive
  // combination of isBeforeInBlock and moveBefore calls (which can be O(n²))
  DenseMap<Operation *, unsigned> opOrder;
  block->walk([&](Operation *op) { opOrder.insert({op, opOrder.size()}); });
  SmallVector<StorageGetOp> getters;
  for (auto [result, storage, offset] : gettersToCreate) {
    SmallDenseMap<Block *, StorageGetOp> getterForBlock;
    for (auto *user : llvm::make_early_inc_range(result.getUsers())) {
      auto &getter = getterForBlock[user->getBlock()];
      // Create a local getter in front of each user, except for
      // `AllocStorageOp`s, for which we create a block-wider accessor.
      auto userOrder = opOrder.lookup(user);
      if (!getter || !result.getDefiningOp<AllocStorageOp>()) {
        ImplicitLocOpBuilder builder(result.getLoc(), user);
        getter =
            StorageGetOp::create(builder, result.getType(), storage, offset);
        getters.push_back(getter);
        opOrder[getter] = userOrder;
      } else if (userOrder < opOrder.lookup(getter)) {
        getter->moveBefore(user);
        opOrder[getter] = userOrder;
      }
      user->replaceUsesOfWith(result, getter);
    }
  }

  // Create the substorage accessor at the beginning of the block.
  Operation *storageOwner = storage.getDefiningOp();
  if (!storageOwner)
    storageOwner = cast<BlockArgument>(storage).getOwner()->getParentOp();

  if (storageOwner->isProperAncestor(block->getParentOp())) {
    auto substorage = AllocStorageOp::create(
        builder, block->getParentOp()->getLoc(),
        StorageType::get(&getContext(), currentByte), storage);
    for (auto *op : ops)
      op->replaceUsesOfWith(storage, substorage);
    for (auto op : getters)
      op->replaceUsesOfWith(storage, substorage);
  } else {
    storage.setType(StorageType::get(&getContext(), currentByte));
  }
}
