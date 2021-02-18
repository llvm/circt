//===- AlwaysFusion.cpp - Always Fusion Pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs simple fusion of always_ff in the same
// region.
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Visitors.h"

using namespace circt;
using namespace sv;

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
    assert(isa<YieldOp>(block1.back()) &&
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

struct AlwaysFusionPass : public AlwaysFusionBase<AlwaysFusionPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void AlwaysFusionPass::runOnOperation() {
  auto *op = getOperation();

  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  bool graphChanged = false;

  auto kindInterface = dyn_cast<mlir::RegionKindInterface>(op);
  for (unsigned i = 0; i < op->getNumRegions(); ++i) {
    // If the operation does not implement the region kind interface, all of
    // its regions are implicitly regular SSACFG region. Since we are blindly
    // combining alwaysff blocks, make sure they are in a graph region.
    if (!kindInterface || kindInterface.getRegionKind(i) == RegionKind::SSACFG)
      continue;

    if (op->getRegion(i).empty())
      continue;

    // Graph regions only have 1 block.
    auto &block = op->getRegion(i).front();

    // A set of operations in the current block which are mergable. Any
    // operation in this set is a candidate for another similar operation to
    // merge in to.
    DenseSet<Operation *, SimpleOperationInfo> foundOps;
    for (sv::AlwaysFFOp alwaysOp :
         llvm::make_early_inc_range(block.getOps<sv::AlwaysFFOp>())) {
      // check if we have encountered an equivalent operation already.  If we
      // have, merge them together and delete the old operation.
      auto itAndInserted = foundOps.insert(alwaysOp);
      if (!itAndInserted.second) {
        // Merge with a similar alwaysff
        mergeOperations(*itAndInserted.first, alwaysOp);
        alwaysOp.erase();
        graphChanged = true;
      }
    }
  }

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!graphChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createAlwaysFusionPass() {
  return std::make_unique<AlwaysFusionPass>();
}
