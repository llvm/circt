//===- GroupResetsAndEnables.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-group-resets-and-enables"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

class ResetGroupingPattern : public RewritePattern {
public:
  using RewritePattern::RewritePattern;
  ResetGroupingPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(ClockTreeOp::getOperationName(), benefit, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ClockTreeOp clockTreeOp = dyn_cast<ClockTreeOp>(*op);

    // Group similar resets into single IfOps
    // Create a list of reset values and map from them to the states they reset
    llvm::MapVector<mlir::Value, SmallVector<scf::IfOp>> resetMap;
    auto ifOps = clockTreeOp.getBody().getOps<scf::IfOp>();

    for (auto ifOp : ifOps)
      resetMap[ifOp.getCondition()].push_back(ifOp);

    // Combine IfOps
    bool changed = false;
    for (auto [cond, oldOps] : resetMap) {
      if (oldOps.size() > 1) {
        auto *iteratorStart = oldOps.begin();
        scf::IfOp firstOp = *(iteratorStart++);
        for (auto *thisOp = iteratorStart; thisOp != oldOps.end(); thisOp++) {
          // Inline the before and after region inside the original If
          rewriter.eraseOp(thisOp->thenBlock()->getTerminator());
          rewriter.inlineBlockBefore(thisOp->thenBlock(),
                                     firstOp.thenBlock()->getTerminator());
          // Check we're not inlining an empty block
          if (!thisOp->elseBlock()->empty()) {
            rewriter.eraseOp(thisOp->elseBlock()->getTerminator());
            rewriter.inlineBlockBefore(thisOp->elseBlock(),
                                       firstOp.elseBlock()->getTerminator());
          }
          rewriter.eraseOp(*thisOp);
          changed = true;
        }
      }
    }
    if (!changed)
      return failure();
    return success();
  }
};

class EnableGroupingPattern : public RewritePattern {
public:
  using RewritePattern::RewritePattern;
  EnableGroupingPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(ClockTreeOp::getOperationName(), benefit, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ClockTreeOp clockTreeOp = dyn_cast<ClockTreeOp>(*op);

    // Generate list of blocks to amass StateWrite enables in - these are accompanied by a boolean that dictates whether the body has a terminator to save on an unnecessary trait check later 
    SmallVector<std::pair<Block *, bool>> groupingBlocks;
    groupingBlocks.push_back(std::pair(&clockTreeOp.getBodyBlock(), false));
    for (auto ifOp : clockTreeOp.getBody().getOps<scf::IfOp>()) {
      groupingBlocks.push_back(std::pair(ifOp.thenBlock(), true));
      groupingBlocks.push_back(std::pair(ifOp.elseBlock(), true));
    }

    bool changed = false;
    for (auto [block, hasTerminator]: groupingBlocks) {
      llvm::MapVector<mlir::Value, SmallVector<StateWriteOp>> enableMap;
      auto writeOps = block->getOps<StateWriteOp>();
      for (auto writeOp : writeOps) {
        if (writeOp.getCondition())
          enableMap[writeOp.getCondition()].push_back(writeOp);
      }
      for (auto [enable, writeOps] : enableMap) {
        // Only group if multiple writes share a reset
        if (writeOps.size() > 1) {
            if (hasTerminator) {
              rewriter.setInsertionPoint(block->getTerminator());
            } else {
              rewriter.setInsertionPointToEnd(block);
            }
          scf::IfOp ifOp = rewriter.create<scf::IfOp>(rewriter.getUnknownLoc(),
                                                      enable, false);
          for (auto writeOp : writeOps) {
            rewriter.updateRootInPlace(writeOp, [&]() {
              writeOp->moveBefore(ifOp.thenBlock()->getTerminator());
              writeOp.getConditionMutable().erase(0);
            });
          }
          changed = true;
        }
      }
    }
    if (!changed)
      return failure();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct GroupResetsAndEnablesPass
    : public GroupResetsAndEnablesBase<GroupResetsAndEnablesPass> {
  GroupResetsAndEnablesPass() = default;
  GroupResetsAndEnablesPass(const GroupResetsAndEnablesPass &pass)
      : GroupResetsAndEnablesPass() {}

  void runOnOperation() override;
  LogicalResult runOnModel(ModelOp modelOp);
};
} // namespace

void GroupResetsAndEnablesPass::runOnOperation() {
  for (auto op : llvm::make_early_inc_range(getOperation().getOps<ModelOp>()))
    if (failed(runOnModel(op)))
      return signalPassFailure();
}

LogicalResult GroupResetsAndEnablesPass::runOnModel(ModelOp modelOp) {
  LLVM_DEBUG(llvm::dbgs() << "Grouping resets and enables in `"
                          << modelOp.getName() << "`\n");

  MLIRContext &context = getContext();
  RewritePatternSet patterns(&context);
  patterns.insert<ResetGroupingPattern>(1, &context);
  patterns.insert<EnableGroupingPattern>(1, &context);
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  (void)applyPatternsAndFoldGreedily(modelOp, std::move(patterns), config);
  return success();
}

std::unique_ptr<Pass> arc::createGroupResetsAndEnablesPass() {
  return std::make_unique<GroupResetsAndEnablesPass>();
}
