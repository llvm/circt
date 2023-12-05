//===- LayerSink.cpp - Sink ops into layer blocks -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A module pass that sinks operations into layer blocks.
//
// This works by iterating, in reverse topological order, over the strongly
// connected components (SCC) for operations in a FIRRTL module.  The operations
// in each SCC are iterated over to compute the "highest" layer that they can be
// sunk into.  This is determined by analyzing the users of all operations in
// each SCC.  Following that, all operations are moved into the highest layer.
//
// SCC iteration relies on the graph traits defined in FIRRTLConnectionGraph.
// Currently this is deficient for partial writes to aggregates, i.e.,
// FConnectLike operations where the destination is a subaccess, subfield, or
// subindex.  For situations involviung these operations, this pass will not
// fail, but will be less effective than it should be.
//
// Tests tracking improvements related to aggregates are located here:
//
//     test/Dialect/FIRRTL/layer-sink-xfail.mlir
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLConnectionGraph.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-layer-sink"

using namespace circt;
using namespace firrtl;

using circt::firrtl::detail::FIRRTLOperation;
using llvm::EquivalenceClasses;

namespace {
/// A control-flow sink pass.
class LayerSink : public LayerSinkBase<LayerSink> {
  void runOnOperation() override;

  /// This contains disjoint sets of all operations in a module that either
  /// already are in a layer block or can be safely moved into a layer block if
  /// all operations in this set are also moved.  Operations which should not be
  /// moved are NOT added to an equivalence class!
  EquivalenceClasses<Operation *> layerSets;

  /// A mapping of a leader from the layerSets to the parent in which this
  /// exists or can be moved.
  DenseMap<Operation *, Operation *> leaderToParent;

  /// A mapping of a layer/parent to the leader which contains all the
  /// operations that are under or can be moved under this layer/parent;
  DenseMap<Operation *, Operation *> parentToLeader;

  /// Return the parent of two operations.  `Initial` is the current "best"
  /// operation (and maye be null if no operation is known to be best) and `x`
  /// is the new operation (and must not be null).  This will never return an
  /// operation above an `FModuleOp`.
  Operation *foldParent(Operation *initial, Operation *x);

  /// Return the common parent of all operations in an SCC.  If this returns
  /// `nullptr` then it indicates that there is no information to indicate a
  /// better common parent exists.
  Operation *processScc(llvm::scc_iterator<FIRRTLOperation *> &i);
};
} // namespace

Operation *LayerSink::foldParent(Operation *initial, Operation *x) {
  if (llvm::isa_and_nonnull<FModuleOp>(initial))
    return initial;

  assert(x && "x must not be null");
  if (llvm::isa<FModuleOp>(x))
    return x;

  // Determine the parent of `x`.  This is either the actual parent or, if this
  // exists in layerSets, then it is the layer that `x` will be moved into.
  Operation *xParent = x->getParentOp();
  auto leader = layerSets.findLeader(x);
  if (leader != layerSets.member_end())
    xParent = leaderToParent.lookup(*leader);

  if (!initial)
    return xParent;

  while (initial != xParent)
    initial = initial->getParentOp();

  return initial;
}

Operation *LayerSink::processScc(llvm::scc_iterator<FIRRTLOperation *> &i) {

  // Mutable temporary that will be updated with the current "best parent".  The
  // initial value of `nullptr` indicates that there is no relevant information
  // indicating a change of parent should occur.
  Operation *bestParent = nullptr;

  // Cache information about what operations are in the SCC.  This is queried to
  // know when a user is in the SCC and analysis of that user contains no
  // bearing on the parent.
  llvm::DenseSet<Operation *> sccOps(i->begin(), i->end());

  for (auto *op : *i) {
    // If the parent is already known to be a module, then stop processing this
    // SCC---continued processing cannot change this result.
    if (isa_and_nonnull<FModuleOp>(bestParent))
      break;

    // Process connect destinations only.
    if (auto connect = dyn_cast<FConnectLike>(op)) {
      auto dest = connect.getDest();
      if (auto blockArg = dyn_cast<BlockArgument>(dest)) {
        bestParent = blockArg.getDefiningOp();
        continue;
      }
      auto *definingOp = dest.getDefiningOp();
      if (sccOps.contains(definingOp))
        continue;
      bestParent = foldParent(bestParent, definingOp);
      continue;
    }

    // If this operation is marked don't touch, then treat it as having itself
    // as a user.  This has the effect of causing it to not be moved.
    if (hasDontTouch(op)) {
      bestParent = foldParent(bestParent, op);
      continue;
    }

    // Process operation users not in this SCC.  Skip connect destinations.
    for (auto &use : op->getUses()) {
      auto *user = use.getOwner();
      // Skip users in the same SCC.
      if (sccOps.contains(user))
        continue;
      // Skip uses that are connect desintations.
      if (auto connect = dyn_cast<FConnectLike>(user))
        if (use.get() == connect.getDest())
          continue;
      bestParent = foldParent(bestParent, user);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "SCC:\n"
                 << "  operations:\n";
    for (auto *op : *i)
      llvm::dbgs() << "    - op: " << *op << "\n";
    llvm::dbgs() << "  commonParent: ";
    if (auto moduleOp = dyn_cast_or_null<FModuleOp>(bestParent))
      llvm::dbgs() << "module";
    else if (auto layerBlockOp = dyn_cast_or_null<LayerBlockOp>(bestParent))
      llvm::dbgs() << layerBlockOp.getLayerName();
    else {
      if (bestParent)
        llvm::dbgs() << *bestParent << "\n";
      assert(!bestParent && "expected commonParent to be null");
      llvm::dbgs() << "<null>";
    }
    llvm::dbgs() << "\n";
  });

  assert((!bestParent || isa<FModuleOp>(bestParent) ||
          isa<LayerBlockOp>(bestParent)) &&
         "expected parent to be null, a Module, or a LayerBlock");
  return bestParent;
}

void LayerSink::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LayerSink "
                      "---------------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);

  FIRRTLOperation *moduleOp = getOperation();

  // Iterate over the SCCs in this module, populating `layerSets` with
  // operations which should be moved.
  for (llvm::scc_iterator<detail::FIRRTLOperation *>
           i = llvm::scc_begin(moduleOp),
           e = llvm::scc_end(moduleOp);
       i != e; ++i) {

    // Determine the best parent for all operations in this SCC.  If this is
    // null, then it means there is no information indicating that the
    // operations can be moved.
    Operation *bestParent = processScc(i);

    // Do not add anything which shouldn't be moved to `layerSets`.
    if (!bestParent || isa<FModuleOp>(bestParent))
      continue;

    // Update `layerSets` with the operations in this SCC.  If this is a new
    // equivalence class, then update `leaderToParent`/`parentToLeader` and
    // start a new equivalence class.  Otherwise, merge these operations into an
    // existing equivalence class.
    auto *first = *(*i).begin();
    auto leaderIt = parentToLeader.find(bestParent);
    if (leaderIt == parentToLeader.end()) {
      leaderToParent.insert({first, bestParent});
      parentToLeader.insert({bestParent, first});
      leaderIt = parentToLeader.find(bestParent);
      layerSets.insert(first);
    } else
      layerSets.unionSets(leaderIt->second, first);

    for (auto *op : *i)
      layerSets.unionSets(leaderIt->second, op);
  }

  // Iterate over each `layerSet` and move all operations into their new layer.
  // Do this while preserving the original order in the module.
  for (auto leaderIt = layerSets.begin(), leaderEnd = layerSets.end();
       leaderIt != leaderEnd; ++leaderIt) {
    if (!leaderIt->isLeader())
      continue;

    SmallVector<Operation *> layerOpsReverseModuleOrdered(
        layerSets.member_begin(leaderIt), layerSets.member_end());
    llvm::sort(layerOpsReverseModuleOrdered.begin(),
               layerOpsReverseModuleOrdered.end(),
               [](Operation *a, Operation *b) {
                 auto aBlock = a->getBlock(), bBlock = b->getBlock();
                 if (aBlock == bBlock)
                   return !a->isBeforeInBlock(b);
                 while (isa<FModuleOp>(aBlock->getParentOp())) {
                   aBlock = a->getParentOp()->getBlock();
                   if (aBlock == bBlock)
                     return true;
                 }
                 return false;
               });

    auto commonParent = leaderToParent.find(leaderIt->getData());
    assert(commonParent != leaderToParent.end() &&
           "a common parent should already exist in leaderToParent");
    assert(commonParent->second && "a common parent should not be null");

    for (auto *op : layerOpsReverseModuleOrdered) {
      auto &region = commonParent->second->getRegion(0);
      op->moveBefore(&region.front(), region.front().begin());
    }
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLayerSinkPass() {
  return std::make_unique<LayerSink>();
}
