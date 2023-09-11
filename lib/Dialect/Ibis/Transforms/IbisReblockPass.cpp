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

#include <iterator>

using namespace circt;
using namespace ibis;

namespace {

struct ReblockPass : public IbisReblockBase<ReblockPass> {
  void runOnOperation() override;

  LogicalResult reblock(Block &block, DictionaryAttr blockInfo);
};
} // anonymous namespace

void ReblockPass::runOnOperation() {
  MethodOp parent = getOperation();
  auto &region = parent.getRegion();
  // Fetch the 'ibis.blockinfo' attribute from the parent operation. This will
  // contain information tagged to each MLIR block which needs to be propagated
  // to the corresponding ibis.block operations.
  auto blockInfoAttr = parent->getAttrOfType<DictionaryAttr>("ibis.blockinfo");

  if (!blockInfoAttr) {
    parent->emitOpError("missing 'ibis.blockinfo' attribute");
    return signalPassFailure();
  }

  if (!isRegionSSAMaximized(region)) {
    parent->emitOpError() << "region is not in maximal SSA form";
    return signalPassFailure();
  }

  auto blockID = [&](Block &block) {
    return std::distance(region.front().getIterator(), block.getIterator());
  };

  for (Block &block : region.getBlocks()) {
    DictionaryAttr blockInfo =
        blockInfoAttr.getAs<DictionaryAttr>(Twine(blockID(block)).str());

    if (!blockInfo) {
      parent->emitOpError("missing 'ibis.blockinfo' attribute for block " +
                          Twine(blockID(block)));
      return signalPassFailure();
    }

    if (failed(reblock(block, blockInfo)))
      return signalPassFailure();
  }
}

LogicalResult ReblockPass::reblock(mlir::Block &block,
                                   DictionaryAttr blockInfo) {
  // Record whether an ibis.call was found in this block
  bool hadCall = false;
  llvm::SmallVector<Operation *> blockOps;
  Operation *terminator = block.getTerminator();
  for (Operation &op : block.getOperations()) {
    if (&op == terminator)
      continue;

    hadCall |= isa<CallOp>(op);
    blockOps.push_back(&op);
  }

  if (hadCall && blockOps.size() > 1)
    return block.getParentOp()->emitError(
        "a block has both calls and other ops");

  if (hadCall) {
    // Nothing to do - call ops are themselves considered control-passing
    // operations.
    return success();
  }

  // Given that we're in maximum SSA form, the only values we need to return
  // from within the block are values used by the terminator, if they're not
  // already block arguments.

  // Maintain a mapping between the value used by the terminator and the
  // OpOperands that holds said value.
  llvm::MapVector<Value, llvm::SmallVector<OpOperand *>> terminatorOperands;
  for (auto &operand : terminator->getOpOperands()) {
    if (operand.get().isa<BlockArgument>())
      continue;
    terminatorOperands[operand.get()].push_back(&operand);
  }

  // The above de-aliased multiple-returns of the same value. We now gather
  // the set of types that needs to be returned from within the block.
  llvm::SmallVector<Type> blockRetTypes;
  llvm::SmallVector<Value> blockReturnValues;
  for (auto &[v, _] : terminatorOperands) {
    blockReturnValues.push_back(v);
    blockRetTypes.push_back(v.getType());
  }

  auto b = OpBuilder::atBlockBegin(&block);

  // Lookup block location from the block info.
  auto loc = blockInfo.getAs<LocationAttr>("loc");
  if (!loc) {
    block.getParentOp()->emitError("missing 'loc' attribute for block");
    return failure();
  }
  auto ibisBlock = b.create<BlockOp>(loc, blockRetTypes, ValueRange{});

  // Move operations into the `ibis.block` op.
  BlockReturnOp blockReturn =
      cast<BlockReturnOp>(ibisBlock.getBodyBlock()->getTerminator());

  for (auto *op : blockOps)
    op->moveBefore(blockReturn);

  // Append the terminator operands to the block return.
  blockReturn->setOperands(blockReturnValues);

  // Replace the basic block terminator operands with the block return values.
  for (auto [blockRet, termOperands] :
       llvm::zip(ibisBlock.getResults(), terminatorOperands)) {
    for (auto *termOperand : termOperands.second)
      termOperand->set(blockRet);
  }

  return success();
}

std::unique_ptr<Pass> circt::ibis::createReblockPass() {
  return std::make_unique<ReblockPass>();
}
