//===- ArcDeadCodeElimination.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-dce"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_ARCDEADCODEELIMINATION
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

namespace {
struct ArcDeadCodeEliminationPass
    : public impl::ArcDeadCodeEliminationBase<ArcDeadCodeEliminationPass> {
  void runOnOperation() override;
};
} // namespace

void ArcDeadCodeEliminationPass::runOnOperation() {
  // Initially mark all side-effecting ops as live.
  LLVM_DEBUG(llvm::dbgs() << "Marking initially live ops\n");
  DenseSet<Operation *> liveOps;
  SetVector<Operation *> worklist;
  SmallVector<Operation *, 0> maybeDeadOps;
  getOperation().walk([&](Operation *op) {
    if (isa<StateOp, CallOp>(op) && wouldOpBeTriviallyDead(op)) {
      maybeDeadOps.push_back(op);
      return;
    }
    liveOps.insert(op);
    for (auto operand : op->getOperands())
      if (auto *defOp = operand.getDefiningOp())
        worklist.insert(defOp);
  });
  LLVM_DEBUG(llvm::dbgs() << "- Found " << liveOps.size()
                          << " initially live ops\n");
  LLVM_DEBUG(llvm::dbgs() << "- Worklist initially has " << worklist.size()
                          << " ops\n");

  // Mark operations as live if any of their users are live.
  LLVM_DEBUG(llvm::dbgs() << "Analyzing liveness\n");
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();

    // If the op is already marked live, there's nothing to do.
    if (liveOps.contains(op))
      continue;

    // If the op would be trivially dead and thus deletable if it had no users,
    // check if all users are dead. If they are, don't mark this op as live.
    if (wouldOpBeTriviallyDead(op)) {
      auto allUsersDead = llvm::all_of(
          op->getUsers(), [&](auto *user) { return !liveOps.contains(user); });
      if (allUsersDead)
        continue;
    }

    // Otherwise mark the op as live, and add all its operands to the worklist,
    // since they will become live as well.
    liveOps.insert(op);
    for (auto operand : op->getOperands())
      if (auto *defOp = operand.getDefiningOp())
        if (!liveOps.contains(defOp))
          worklist.insert(defOp);
  }

  // Go through the list of potentially dead ops and remove the ones that ended
  // up being live. That leaves only the dead ops in the list.
  llvm::erase_if(maybeDeadOps, [&](auto *op) { return liveOps.contains(op); });
  LLVM_DEBUG(llvm::dbgs() << "- " << liveOps.size() << " live\n");
  LLVM_DEBUG(llvm::dbgs() << "- " << maybeDeadOps.size() << " dead\n");

  // Delete the dead ops. This may form cycles, so drop all references first
  // before actually erasing the ops.
  for (auto *op : maybeDeadOps)
    op->dropAllReferences();
  for (auto *op : maybeDeadOps)
    op->erase();
  numUsesRemoved += maybeDeadOps.size();
  LLVM_DEBUG(llvm::dbgs() << "Removed " << maybeDeadOps.size()
                          << " dead ops\n");
}

std::unique_ptr<mlir::Pass> arc::createArcDeadCodeEliminationPass() {
  return std::make_unique<ArcDeadCodeEliminationPass>();
}
