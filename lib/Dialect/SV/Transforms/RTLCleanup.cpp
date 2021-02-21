//===- RTLCleanup.cpp - RTL Cleanup Pass ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs various cleanups and canonicalization
// transformations for rtl.module bodies.
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

namespace {

/// Check the equivalence of operations by doing a deep comparison of operands
/// and attributes, but does not compare the content of any regions attached to
/// each op.
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return mlir::OperationEquivalence::computeHash(
        const_cast<Operation *>(opC));
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return mlir::OperationEquivalence::isEquivalentTo(lhs, rhs);
  }
};

} // end anonymous namespace

// Merge two regions together. These regions must only have a one block.
static void mergeRegions(Region *region1, Region *region2) {
  assert(region1->getBlocks().size() <= 1 && region2->getBlocks().size() <= 1 &&
         "Can only merge regions with a single block");
  if (region1->empty()) {
    // If both regions are empty, move on to the next pair of regions
    if (region2->empty())
      return;
    // If the first region has no block, move the second region's block over.
    region1->getBlocks().splice(region1->end(), region2->getBlocks());
    return;
  }
  // If the second region is not empty, splice its block into the end of the
  // first region.
  if (!region2->empty()) {
    auto &block1 = region1->front();
    auto &block2 = region2->front();
    // Remove the terminator from the first block before merging.
    assert(isa<sv::YieldOp>(block1.back()) &&
           "Block should be terminated by an sv.yield operation");
    block1.back().erase();
    block1.getOperations().splice(block1.end(), block2.getOperations());
  }
}

/// Inline all regions from the second operation into the first.
static void mergeOperations(Operation *op1, Operation *op2) {
  assert(op1 != op2 && "Cannot merge an op into itself");
  assert(op1->getNumRegions() == 2 &&
         "alwaysff should always have two regions");
  for (unsigned i = 0; i != 2; ++i)
    mergeRegions(&op1->getRegion(i), &op2->getRegion(i));
}

//===----------------------------------------------------------------------===//
// AlwaysFusionPass
//===----------------------------------------------------------------------===//

namespace {
struct RTLCleanupPass : public sv::RTLCleanupBase<RTLCleanupPass> {
  void runOnOperation() override;
  void runOnRegion(Region &region);

private:
  bool anythingChanged;
};
} // end anonymous namespace

void RTLCleanupPass::runOnOperation() {
  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;
  runOnRegion(getOperation().getBody());

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

void RTLCleanupPass::runOnRegion(Region &region) {
  if (region.getBlocks().size() != 1)
    return;
  Block &body = region.front();

  // A set of operations in the current block which are mergable. Any
  // operation in this set is a candidate for another similar operation to
  // merge in to.
  DenseSet<Operation *, SimpleOperationInfo> alwaysFFOpsSeen;
  llvm::SmallDenseMap<Attribute, Operation *, 4> ifdefOps;

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Recursively process any regions in the op before we visit it.
    for (size_t i = 0, e = op.getNumRegions(); i != e; ++i)
      runOnRegion(op.getRegion(i));

    // Merge alwaysff operations by hashing them to check to see if we've
    // already encountered one.  If so, merge them and reprocess the body.
    if (auto alwaysOp = dyn_cast<sv::AlwaysFFOp>(op)) {
      // Merge identical alwaysff's together and delete the old operation.
      auto itAndInserted = alwaysFFOpsSeen.insert(alwaysOp);
      if (itAndInserted.second)
        continue;
      auto *existingAlways = *itAndInserted.first;

      mergeOperations(existingAlways, alwaysOp);
      alwaysOp.erase();
      anythingChanged = true;

      // Reprocess the merged body because this may have uncovered other
      // simplifications.
      runOnRegion(existingAlways->getRegion(0));
      runOnRegion(existingAlways->getRegion(1));
      continue;
    }

    // Merge graph ifdefs anywhere in the module.
    if (auto ifdefOp = dyn_cast<sv::IfDefOp>(op)) {
      auto *&entry = ifdefOps[ifdefOp.condAttr()];
      if (!entry) {
        entry = ifdefOp;
        continue;
      }

      mergeOperations(entry, ifdefOp);
      ifdefOp.erase();
      anythingChanged = true;

      // TODO: Reprocess the merged body because this may have uncovered other
      // simplifications.
      runOnRegion(entry->getRegion(0));
      runOnRegion(entry->getRegion(1));
      continue;
    }

    // TODO: Merge procedural ifdef's with neighboring ones of the same
    // condition.
    // TODO: Merge procedural if's.
  }
}

std::unique_ptr<Pass> circt::sv::createRTLCleanupPass() {
  return std::make_unique<RTLCleanupPass>();
}
