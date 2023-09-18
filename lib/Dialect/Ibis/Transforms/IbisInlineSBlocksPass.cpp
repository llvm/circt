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

static size_t getNumOps(mlir::Block *block) {
  return std::distance(block->begin(), block->end());
}

struct InlineSBlocksPass : public IbisInlineSBlocksBase<InlineSBlocksPass> {
  void runOnOperation() override;

  // Inlines an ibis.sblock into the CFG of the parent block.
  // Will attempt to reduce the number of empty blocks (i.e. blocks only
  // containing branches to other blocks) - this is imperative, since region
  // simplification is not allowed to run after this pass.
  void inlineSBlock(OpBuilder &b, ibis::StaticBlockOp op,
                    DenseMap<Block *, DictionaryAttr> &blockAttrMap);
};

class InlineSBlocksPattern : public OpConversionPattern<ibis::StaticBlockOp> {
public:
  InlineSBlocksPattern(MLIRContext *context,
                       DenseMap<Block *, DictionaryAttr> &blockAttrMap)
      : OpConversionPattern<ibis::StaticBlockOp>(context),
        blockAttrMap(blockAttrMap) {}
  using OpAdaptor =
      typename OpConversionPattern<ibis::StaticBlockOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ibis::StaticBlockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
      rewriter.setInsertionPointToEnd(predBlock);
    } else {
      rewriter.setInsertionPointToEnd(predBlock);
      rewriter.create<cf::BranchOp>(loc, sblockBody);
      // Outgoing control flow should start from the end of the sblock body.
      rewriter.setInsertionPointToEnd(sblockBody);
    }

    // Create control flow from the sblock body to the successor block.
    if (getNumOps(succBlock) == 1) {
      // Successor block only contains a branch, so we can merge it with the
      // sblock body.
      sblockBody->getOperations().splice(sblockBody->end(),
                                         succBlock->getOperations());
      rewriter.eraseBlock(succBlock);
    } else {
      // Successor block contains more than just a branch, so we need to create
      // a branch to it.
      rewriter.create<cf::BranchOp>(loc, succBlock);
    }

    // Add an attribute for the source block location.
    llvm::SmallVector<NamedAttribute> blockAttrs;
    blockAttrs.push_back(
        rewriter.getNamedAttr("loc", static_cast<LocationAttr>(loc)));
    if (auto maxThreads = op.getMaxThreadsAttr())
      blockAttrs.push_back(rewriter.getNamedAttr("maxThreads", maxThreads));

    blockAttrMap[sblockBody] = rewriter.getDictionaryAttr(blockAttrs);
    rewriter.eraseOp(op);
    return success();
  }

  DenseMap<Block *, DictionaryAttr> &blockAttrMap;
};

} // anonymous namespace

void InlineSBlocksPass::runOnOperation() {
  MethodOp parent = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<cf::ControlFlowDialect>();
  target.addIllegalOp<ibis::StaticBlockOp>();

  DenseMap<Block *, DictionaryAttr> blockAttrMap;
  RewritePatternSet patterns(&getContext());
  patterns.add<InlineSBlocksPattern>(&getContext(), blockAttrMap);

  if (failed(applyPartialConversion(parent, target, std::move(patterns))))
    return signalPassFailure();

  // Add block annotations to the parent method op.
  llvm::SmallVector<NamedAttribute> blockInfo;
  for (auto [block, attrs] : blockAttrMap)
    blockInfo.push_back(NamedAttribute(
        StringAttr::get(&getContext(), std::to_string(blockID(block))), attrs));

  parent->setAttr("ibis.blockinfo",
                  DictionaryAttr::get(&getContext(), blockInfo));
}

std::unique_ptr<Pass> circt::ibis::createInlineSBlocksPass() {
  return std::make_unique<InlineSBlocksPass>();
}
