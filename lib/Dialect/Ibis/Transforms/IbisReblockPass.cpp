//===- IbisReblockPass.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace ibis;

namespace {

struct ReblockPass : public IbisReblockBase<ReblockPass> {
  void runOnOperation() override;

  // Transforms an `ibis.sblock.inline.begin/end` scope into an `ibis.sblock`.
  LogicalResult reblock(ibis::InlineStaticBlockBeginOp);
};
} // anonymous namespace

void ReblockPass::runOnOperation() {
  MethodOp parent = getOperation();

  for (auto blockBeginOp : llvm::make_early_inc_range(
           parent.getOps<ibis::InlineStaticBlockBeginOp>())) {
    if (failed(reblock(blockBeginOp)))
      return signalPassFailure();
  }
}

LogicalResult ReblockPass::reblock(ibis::InlineStaticBlockBeginOp beginOp) {
  // Determine which values we need to return from within the block scope.
  // This is done by collecting all values defined within the start/end scope,
  // and recording uses that exist outside of the scope.
  ibis::InlineStaticBlockEndOp endOp = beginOp.getEndOp();
  assert(endOp);

  auto startIt = beginOp->getIterator();
  auto endIt = endOp->getIterator();
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
  for (auto &op : llvm::make_range(startIt, endIt)) {
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
  auto ibisBlock =
      b.create<StaticBlockOp>(beginOp.getLoc(), blockRetTypes, ValueRange{});

  // The new `ibis.sblock` should inherit the attributes of the block begin op.
  ibisBlock->setAttrs(beginOp->getAttrs());

  // Move operations into the `ibis.sblock` op.
  BlockReturnOp blockReturn =
      cast<BlockReturnOp>(ibisBlock.getBodyBlock()->getTerminator());

  for (auto &op : llvm::make_early_inc_range(llvm::make_range(startIt, endIt)))
    op.moveBefore(blockReturn);

  // Append the terminator operands to the block return.
  blockReturn->setOperands(blockRetValues);

  // Replace the uses of the returned values outside of the block with the
  // block return values.
  for (auto [blockRet, innerDefAndUses] :
       llvm::zip(ibisBlock.getResults(), returnValueUses)) {
    auto &uses = std::get<1>(innerDefAndUses);
    for (OpOperand *use : uses)
      use->set(blockRet);
  }
  // Erase the start/end ops.
  beginOp.erase();
  endOp.erase();
  return success();
}

std::unique_ptr<Pass> circt::ibis::createReblockPass() {
  return std::make_unique<ReblockPass>();
}
