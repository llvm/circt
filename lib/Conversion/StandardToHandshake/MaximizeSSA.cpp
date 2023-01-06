//===- MaximizeSSA.cpp - SSA Maximization Pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the SSA maximization pass as well as utilities
// for converting a function with standard control flow into maximal SSA form.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

#include <stack>

using namespace mlir;
using namespace circt;

static Block *getDefiningBlock(Value value) {
  // Value is either a block argument...
  if (auto blockArg = dyn_cast<BlockArgument>(value); blockArg)
    return blockArg.getParentBlock();

  // ... or an operation's result
  auto *defOp = value.getDefiningOp();
  assert(defOp);
  return defOp->getBlock();
}

static LogicalResult addArgToTerminator(Block *block, Block *predBlock,
                                        Value value) {

  // Identify terminator branching instruction in predecessor block
  auto branchOp = dyn_cast<BranchOpInterface>(predBlock->getTerminator());
  if (!branchOp)
    return predBlock->getTerminator()->emitError(
        "Expected terminator operation of block to be a "
        "branch-like operation");

  // In the predecessor block's terminator, find all successors that equal
  // the block and add the value to the list of operands it's passed
  auto successors = branchOp->getSuccessors();
  for (uint idx = 0, numSucc = successors.size(); idx < numSucc; ++idx) {
    auto *succBlock = successors[idx];
    if (succBlock == block) {
      branchOp.getSuccessorOperands(idx).append(value);
    }
  }

  return success();
}

namespace circt {

bool isRegionSSAMaximized(Region &region) {

  // Check whether all operands used within each block are also defined within
  // the same block
  for (auto &block : region.getBlocks())
    for (auto &op : block.getOperations())
      for (auto operand : op.getOperands())
        if (getDefiningBlock(operand) != &block)
          return false;

  return true;
}

LogicalResult maximizeSSA(Value value, PatternRewriter &rewriter) {

  // Identify the basic block in which the value is defined
  Block *defBlock = getDefiningBlock(value);

  // Identify all basic blocks in which the value is used (excluding the
  // value-defining block)
  DenseSet<Block *> blocksUsing;
  for (auto &use : value.getUses()) {
    auto *block = use.getOwner()->getBlock();
    if (block != defBlock)
      blocksUsing.insert(block);
  }

  // Prepare a stack to store the list of basic blocks that must be modified for
  // the value to be in maximum SSA form. At all points, blocksUsing is a
  // non-strict superset of the elements contained in blocksToVisit
  std::stack<Block *> blocksToVisit;
  for (auto *block : blocksUsing) {
    blocksToVisit.push(block);
  }

  // Backtrack from all blocks using the value to the value-defining basic
  // block, adding a new block argument for the value along the way. Keep
  // track of which blocks have already been modified to avoid visiting a block
  // more than once while backtracking (possible due to branching control flow)
  DenseMap<Block *, BlockArgument> blockToArg;
  while (!blocksToVisit.empty()) {
    // Retrieve the basic block at the top of the stack, remove it from the
    // stack, and add it to the list of visited blocks
    auto *block = blocksToVisit.top();
    blocksToVisit.pop();

    // Add an argument to the block to hold the value
    blockToArg[block] =
        block->addArgument(value.getType(), rewriter.getUnknownLoc());

    // In all block predecessors, modify the terminator branching instruction to
    // also pass the value to the block
    for (auto *predBlock : block->getPredecessors()) {

      // Modify the terminator instruction
      if (failed(addArgToTerminator(block, predBlock, value)))
        return failure();

      // Now the predecessor block is using the value, so we must also make sure
      // to visit it
      if (predBlock != defBlock) {
        auto [_, blockNewlyUsing] = blocksUsing.insert(predBlock);
        if (blockNewlyUsing) {
          blocksToVisit.push(predBlock);
        }
      }
    }
  }

  // Replace all uses of the value with the newly added block arguments
  SmallVector<Operation *> users;
  for (auto &use : value.getUses()) {
    auto *owner = use.getOwner();
    if (owner->getBlock() != defBlock)
      users.push_back(owner);
  }
  for (auto *user : users)
    user->replaceUsesOfWith(value, blockToArg[user->getBlock()]);

  return success();
}

LogicalResult maximizeSSA(Operation *op, PatternRewriter &rewriter) {
  // Apply SSA maximization on each of the operation's results
  for (auto res : op->getResults())
    if (failed(maximizeSSA(res, rewriter)))
      return failure();

  return success();
}

LogicalResult maximizeSSA(Block *block, PatternRewriter &rewriter) {
  // Apply SSA maximization on each of the block's arguments
  for (auto arg : block->getArguments())
    if (failed(maximizeSSA(arg, rewriter)))
      return failure();

  // Apply SSA maximization on each of the block's operations
  for (auto &op : block->getOperations())
    if (failed(maximizeSSA(&op, rewriter)))
      return failure();

  return success();
}

LogicalResult maximizeSSA(Region &region, PatternRewriter &rewriter) {
  // Apply SSA maximization on each of the region's block
  for (auto &block : region.getBlocks())
    if (failed(maximizeSSA(&block, rewriter)))
      return failure();

  return success();
}
} // namespace circt

namespace {

struct FuncOpMaxSSAConversion : public OpConversionPattern<func::FuncOp> {

  FuncOpMaxSSAConversion(MLIRContext *ctx) : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult conversionStatus = success();
    rewriter.updateRootInPlace(op, [&] {
      PatternRewriter &baseRewriter = rewriter;
      if (failed(maximizeSSA(op.getRegion(), baseRewriter)))
        conversionStatus = failure();
    });
    return success();
  }
};

struct MaximizeSSAPass : public MaximizeSSABase<MaximizeSSAPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns.add<FuncOpMaxSSAConversion>(ctx);
    ConversionTarget target{*ctx};

    // Check that the function is correctly SSA-maximized after the pattern has
    // been applied
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
      return isRegionSSAMaximized(func.getBody());
    });

    // Each function in the module is turned into maximal SSA form
    // independently of the others. Function signatures are never modified
    // by SSA maximization
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createMaximizeSSAPass() {
  return std::make_unique<MaximizeSSAPass>();
}
} // namespace circt
