//===- GroupMerge.cpp - Merge optional groups together --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass merges group definitions which reference the same group
// declaration.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-group-merge"

using namespace circt;
using namespace firrtl;

namespace {
/// A pass that merges optional groups referencing the same group declaration.
struct GroupMerge : public GroupMergeBase<GroupMerge> {
  void runOnOperation() override;
};
} // namespace

void GroupMerge::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running GroupMerge "
                      "--------------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);

  // Track the last group that we saw which referenced a specific group
  // declaration.  Because this pass operates as a single walk of the IR, it is
  // only ever possible that there is one prior group that references a given
  // group declaration.
  llvm::DenseMap<SymbolRefAttr, GroupOp> priorGroups;

  // Recursively walk GroupOps in the module.  Whenever we see a group, check to
  // see if there is a prior group that references the same declaration.  If
  // not, this group becomes the prior group and we continue.  If there is a
  // prior group, then splice the prior group's body into the beginning of this
  // group and erase the prior group.  This group then becomes the new prior
  // group.
  //
  // The recursive walk will cause nested groups to also be merged.
  auto moduleOp = getOperation();
  mlir::IRRewriter rewriter(moduleOp.getContext());
  moduleOp.walk([&](GroupOp group) {
    auto groupName = group.getGroupName();
    // If we haven't seen this group before, then just insert it into
    // priorGroups.
    auto priorGroupIt = priorGroups.find(groupName);
    if (priorGroupIt == priorGroups.end()) {
      priorGroups[groupName] = group;
      return WalkResult::advance();
    }

    // Merge the prior group's body into this group.  Erase the prior group.
    // This merged group is now the new prior group (and will be merged into the
    // next group if one is found).
    auto &priorGroup = priorGroupIt->getSecond();
    rewriter.inlineBlockBefore(priorGroup.getBody(), group.getBody(),
                               group.getBody()->begin());
    priorGroup->erase();
    priorGroups[groupName] = group;
    numMerged++;
    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGroupMergePass() {
  return std::make_unique<GroupMerge>();
}
