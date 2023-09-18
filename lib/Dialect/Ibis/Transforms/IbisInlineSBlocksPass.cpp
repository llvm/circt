//===- IbisInlineSBlocksPass.cpp ------------------------------------------===//
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iterator>

using namespace mlir;
using namespace circt;
using namespace ibis;

namespace {

struct InlineSBlocksPass : public IbisInlineSBlocksBase<InlineSBlocksPass> {
  void runOnOperation() override;

  // Inlines an ibis.sblock into the CFG of the parent block.
  // Will attempt to reduce the number of empty blocks (i.e. blocks only
  // containing branches to other blocks) - this is imperative, since region
  // simplification is not allowed to run after this pass.
  void inlineSBlock(OpBuilder &b, ibis::StaticBlockOp op,
                    DenseMap<Block *, DictionaryAttr> &blockAttrMap);
};
} // anonymous namespace

void InlineSBlocksPass::runOnOperation() {
  MethodOp parent = getOperation();
  // @mortbopet: Would use a PatternRewriter here, if not for the fact that
  // lowering ibis.sblocks modifies the parent operations' attributes, which
  // would be illegal.
  // In the future, if we want speedup here, we could consider introducing
  // either auxilary information-carrying ops within each block, or look into
  // adding upstream support for block attributes.
  DenseMap<Block *, DictionaryAttr> blockAttrMap;
  OpBuilder b(parent);
  for (auto StaticBlockOp :
       llvm::make_early_inc_range(parent.getOps<ibis::StaticBlockOp>()))
    inlineSBlock(b, StaticBlockOp, blockAttrMap);

  // Add block annotations to the parent method op.
  llvm::SmallVector<NamedAttribute> blockInfo;
  for (auto [block, attrs] : blockAttrMap)
    blockInfo.push_back(b.getNamedAttr(std::to_string(blockID(block)), attrs));

  parent->setAttr("ibis.blockinfo", b.getDictionaryAttr(blockInfo));
}

static size_t getNumOps(mlir::Block *block) {
  return std::distance(block->begin(), block->end());
}

void InlineSBlocksPass::inlineSBlock(
    OpBuilder &b, ibis::StaticBlockOp op,
    DenseMap<Block *, DictionaryAttr> &blockAttrMap) {
  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  Block *predBlock = op->getBlock();
  Block *succBlock =
      predBlock->splitBlock(std::next(op.getOperation()->getIterator()));

  // Move the body of the ibis.sblock before the successor block.
  Block *sblockBody = op.getBodyBlock();
  sblockBody->moveBefore(succBlock);
  BlockReturnOp ret = cast<BlockReturnOp>(sblockBody->getTerminator());

  // Replace the ibis.sblock return values with the values that were defined
  // (returned from) within the sblock body, and erase the return op.
  for (auto [res, val] : llvm::zip(op.getResults(), ret.getRetValues()))
    res.replaceAllUsesWith(val);
  ret.erase();

  // Create control flow from the predecessor block to the sblock body.
  if (getNumOps(predBlock) == 1) {
    // Predecessor block only contained the sblock, so we can merge it
    // with the sblock body.
    predBlock->getOperations().splice(predBlock->end(),
                                      sblockBody->getOperations());
    sblockBody->erase();
    sblockBody = predBlock;

    // Outgoing control flow should start from the end of the merged block.
    b.setInsertionPointToEnd(predBlock);
  } else {
    b.setInsertionPointToEnd(predBlock);
    b.create<cf::BranchOp>(loc, sblockBody);
    // Outgoing control flow should start from the end of the sblock body.
    b.setInsertionPointToEnd(sblockBody);
  }

  // Create control flow from the sblock body to the successor block.
  if (getNumOps(succBlock) == 1) {
    // Successor block only contains a branch, so we can merge it with the
    // sblock body.
    sblockBody->getOperations().splice(sblockBody->end(),
                                       succBlock->getOperations());
    succBlock->erase();
  } else {
    // Successor block contains more than just a branch, so we need to create
    // a branch to it.
    b.create<cf::BranchOp>(loc, succBlock);
  }

  // Add an attribute for the source block location.
  llvm::SmallVector<NamedAttribute> blockAttrs;
  blockAttrs.push_back(b.getNamedAttr("loc", static_cast<LocationAttr>(loc)));
  if (auto maxThreads = op.getMaxThreadsAttr())
    blockAttrs.push_back(b.getNamedAttr("maxThreads", maxThreads));

  blockAttrMap[sblockBody] = b.getDictionaryAttr(blockAttrs);
  op.erase();
}

std::unique_ptr<Pass> circt::ibis::createInlineSBlocksPass() {
  return std::make_unique<InlineSBlocksPass>();
}
