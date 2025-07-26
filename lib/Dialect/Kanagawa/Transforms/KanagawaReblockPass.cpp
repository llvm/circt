//===- KanagawaReblockPass.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWAREBLOCK
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace circt;
using namespace kanagawa;

namespace {

struct ReblockPass
    : public circt::kanagawa::impl::KanagawaReblockBase<ReblockPass> {
  void runOnOperation() override;

  // Transforms an `kanagawa.sblock.inline.begin/end` scope into an
  // `kanagawa.sblock`.
  LogicalResult reblock(ArrayRef<Operation *> ops, Operation *blockTerminator);

  /// Track ops that should be erased.
  SmallVector<Operation *> opsToErase;
};

// Returns true if the given op signal that the existing set of sblock
// operations should be closed.
static bool isSBlockTerminator(Operation *op) {
  return op->hasTrait<OpTrait::IsTerminator>() ||
         isa<kanagawa::InlineStaticBlockEndOp, InlineStaticBlockBeginOp>(op);
}

} // anonymous namespace

void ReblockPass::runOnOperation() {
  MethodOp parent = getOperation();

  llvm::SmallVector<Operation *> opsToBlock;
  for (Block &block : parent.getRegion()) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (isSBlockTerminator(&op)) {
        if (opsToBlock.empty())
          continue;
        if (failed(reblock(opsToBlock, &op)))
          return signalPassFailure();
        opsToBlock.clear();
        if (isa<InlineStaticBlockBeginOp>(op))
          opsToBlock.push_back(&op);
      } else
        opsToBlock.push_back(&op);
    }
  }

  llvm::for_each(opsToErase, [](Operation *op) { op->erase(); });
}

LogicalResult ReblockPass::reblock(ArrayRef<Operation *> ops,
                                   Operation *blockTerminator) {
  // Determine which values we need to return from within the block scope.
  // This is done by collecting all values defined within the start/end scope,
  // and recording uses that exist outside of the scope.
  kanagawa::InlineStaticBlockBeginOp blockBeginOp;
  if (isa<InlineStaticBlockEndOp>(blockTerminator)) {
    blockBeginOp = dyn_cast<InlineStaticBlockBeginOp>(ops.front());
    assert(blockBeginOp &&
           "Expected block begin op when a block end block was provided");
    ops = ops.drop_front();
  }

  Operation *beginOp = ops.front();
  auto startIt = ops.front()->getIterator();
  Operation *endOp = ops.back();
  auto terminatorIt = blockTerminator->getIterator();
  Block *sblockParentBlock = beginOp->getBlock();

  auto usedOutsideBlock = [&](OpOperand &use) {
    Operation *owner = use.getOwner();
    Block *useBlock = owner->getBlock();
    if (useBlock != sblockParentBlock)
      return true;
    bool isBefore = owner->isBeforeInBlock(beginOp);
    bool isAfter = isBefore ? false : endOp->isBeforeInBlock(owner);
    return isBefore || isAfter;
  };

  llvm::MapVector<Value, llvm::SmallVector<OpOperand *>> returnValueUses;
  for (auto &op : llvm::make_range(startIt, terminatorIt)) {
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses())
        if (usedOutsideBlock(use))
          returnValueUses[result].push_back(&use);
    }
  }

  // Gather the set of types that needs to be returned from within the block.
  llvm::SmallVector<Type> blockRetTypes;
  llvm::SmallVector<Value> blockRetValues;
  for (auto &[v, _] : returnValueUses) {
    blockRetTypes.push_back(v.getType());
    blockRetValues.push_back(v);
  }

  auto b = OpBuilder(beginOp);
  auto kanagawaBlock =
      StaticBlockOp::create(b, beginOp->getLoc(), blockRetTypes, ValueRange{});

  // The new `kanagawa.sblock` should inherit the attributes of the block begin
  // op, if provided.
  if (blockBeginOp)
    kanagawaBlock->setAttrs(blockBeginOp->getAttrs());

  // Move operations into the `kanagawa.sblock` op.
  BlockReturnOp blockReturn =
      cast<BlockReturnOp>(kanagawaBlock.getBodyBlock()->getTerminator());

  for (auto &op :
       llvm::make_early_inc_range(llvm::make_range(startIt, terminatorIt)))
    op.moveBefore(blockReturn);

  // Append the terminator operands to the block return.
  blockReturn->setOperands(blockRetValues);

  // Replace the uses of the returned values outside of the block with the
  // block return values.
  for (auto [blockRet, innerDefAndUses] :
       llvm::zip(kanagawaBlock.getResults(), returnValueUses)) {
    auto &uses = std::get<1>(innerDefAndUses);
    for (OpOperand *use : uses)
      use->set(blockRet);
  }

  // If this was an explicit sblock, erase the markers.
  if (blockBeginOp) {
    opsToErase.push_back(blockBeginOp);
    opsToErase.push_back(blockTerminator);
  }

  return success();
}

std::unique_ptr<Pass> circt::kanagawa::createReblockPass() {
  return std::make_unique<ReblockPass>();
}
