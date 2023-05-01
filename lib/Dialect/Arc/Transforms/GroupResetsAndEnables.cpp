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

    for (auto ifOp : clockTreeOp.getBody().getOps<scf::IfOp>())
      resetMap[ifOp.getCondition()].push_back(ifOp);

    // Combine IfOps
    bool changed = false;
    for (auto [cond, oldOps] : resetMap) {
      if (oldOps.size() <= 1)
        continue;
      scf::IfOp lastIfOp = oldOps.pop_back_val();
      for (auto thisOp : oldOps) {
        // Inline the before and after region inside the original If
        rewriter.eraseOp(thisOp.thenBlock()->getTerminator());
        rewriter.inlineBlockBefore(thisOp.thenBlock(),
                                   &lastIfOp.thenBlock()->front());
        // Check we're not inlining an empty block
        if (auto *elseBlock = thisOp.elseBlock()) {
          rewriter.eraseOp(elseBlock->getTerminator());
          rewriter.inlineBlockBefore(elseBlock, &lastIfOp.elseBlock()->front());
        }
        rewriter.eraseOp(thisOp);
        changed = true;
      }
    }
    return success(changed);
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

    // Amass regions that we want to group enables in
    SmallVector<Region *> groupingRegions;
    groupingRegions.push_back(&clockTreeOp.getBody());
    for (auto ifOp : clockTreeOp.getBody().getOps<scf::IfOp>()) {
      groupingRegions.push_back(&ifOp.getThenRegion());
      groupingRegions.push_back(&ifOp.getElseRegion());
    }

    bool changed = false;
    for (auto *region : groupingRegions) {
      llvm::MapVector<mlir::Value, SmallVector<StateWriteOp>> enableMap;
      auto writeOps = region->getOps<StateWriteOp>();
      for (auto writeOp : writeOps) {
        if (writeOp.getCondition())
          enableMap[writeOp.getCondition()].push_back(writeOp);
      }
      for (auto [enable, writeOps] : enableMap) {
        // Only group if multiple writes share an enable
        if (writeOps.size() <= 1)
          continue;
        if (isa<scf::IfOp>(region->getParentOp()))
          rewriter.setInsertionPoint(region->back().getTerminator());
        else
          rewriter.setInsertionPointToEnd(&region->back());
        scf::IfOp ifOp =
            rewriter.create<scf::IfOp>(writeOps[0].getLoc(), enable, false);
        for (auto writeOp : writeOps) {
          rewriter.updateRootInPlace(writeOp, [&]() {
            writeOp->moveBefore(ifOp.thenBlock()->getTerminator());
            writeOp.getConditionMutable().erase(0);
          });
        }
        changed = true;
      }
    }
    return success(changed);
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
  for (auto op : getOperation().getOps<ModelOp>())
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
