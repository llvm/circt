//===- HWCleanup.cpp - HW Cleanup Pass ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs various cleanups and canonicalization
// transformations for hw.module bodies.  This is intended to be used early in
// the HW/SV pipeline to expose optimization opportunities that require global
// analysis.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

namespace {

/// Check the equivalence of operations by doing a deep comparison of operands
/// and attributes, but does not compare the content of any regions attached to
/// each op.
struct AlwaysLikeOpInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return mlir::OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/mlir::OperationEquivalence::directHashValue,
        /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    // Trivially the same.
    if (lhs == rhs)
      return true;
    // Filter out tombstones and empty ops.
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    // Compare attributes.
    if (lhs->getName() != rhs->getName() ||
        lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
        lhs->getNumOperands() != rhs->getNumOperands())
      return false;
    // Compare operands.
    for (auto operandPair : llvm::zip(lhs->getOperands(), rhs->getOperands())) {
      Value lhsOperand = std::get<0>(operandPair);
      Value rhsOperand = std::get<1>(operandPair);
      if (lhsOperand != rhsOperand)
        return false;
    }
    // The two AlwaysOps are similar enough to be combined.
    return true;
  }
};

void eraseOpWithoutUse(Operation *op) {
  if (op && op->use_empty())
    op->erase();
}

// This struct analyses given two conditions. We decompose the atomic values
// (i.e. non AndOp) in the two conditions into three sets `lhsOperands`,
// `rhsOperands` and `commonOperands` so that the following relation holds. This
// struct is used to hoist values in commonOperands outer if op.
//
// lhs = lhsOperands /\ commonOperands
// rhs = rhsOperands /\ commonOperands
struct ConditionPairInformation {
  ConditionPairInformation(Value lhs, Value rhs) : lhs(lhs), rhs(rhs) {
    peelOperands(lhs, lhsOperands);
    peelOperands(rhs, rhsOperands);

    // Take an intersection.
    for (auto operand : lhsOperands)
      if (rhsOperands.contains(operand))
        commonOperand.push_back(operand);

    if (commonOperand.empty())
      return;

    // If lhsOperands are equal to rhsOperands, all conditions are the same.
    if (commonOperand.size() == lhsOperands.size() &&
        commonOperand.size() == rhsOperands.size()) {
      lhsOperands.clear();
      rhsOperands.clear();
      return;
    }

    // Remove all common values from both sets.
    for (auto operand : commonOperand) {
      lhsOperands.remove(operand);
      rhsOperands.remove(operand);
    }
  }

  // Remove ops.
  void cleanUpOps() {
    eraseOpWithoutUse(lhs.getDefiningOp());
    eraseOpWithoutUse(rhs.getDefiningOp());
  }

  Value lhs, rhs;
  // Unique values in the condition lhs and rhs.
  llvm::SmallSetVector<Value, 4> lhsOperands, rhsOperands;
  // Common values in two conditions.
  llvm::SmallVector<Value> commonOperand;

private:
  // Helper function to accmulate operands.
  static void peelOperands(Value cond, llvm::SmallSetVector<Value, 4> &result) {
    auto andOp = dyn_cast_or_null<comb::AndOp>(cond.getDefiningOp());
    if (!andOp) {
      result.insert(cond);
      return;
    }

    llvm::for_each(andOp.getOperands(),
                   [&](Value cond) { result.insert(cond); });
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

  // If the second region is not empty, splice its block into the start of the
  // first region.
  if (!region2->empty()) {
    auto &block1 = region1->front();
    auto &block2 = region2->front();
    block1.getOperations().splice(block1.begin(), block2.getOperations());
  }
}

//===----------------------------------------------------------------------===//
// HWCleanupPass
//===----------------------------------------------------------------------===//

namespace {
struct HWCleanupPass : public sv::HWCleanupBase<HWCleanupPass> {
  HWCleanupPass(bool convertIfToCaseFlag) {
    convertIfToCase = convertIfToCaseFlag;
  }
  void runOnOperation() override;

  void runOnRegionsInOp(Operation &op);
  void runOnGraphRegion(Region &region);
  void runOnProceduralRegion(Region &region);
  void runIfOpsToCaseOnBlock(Block &block);

private:
  /// Inline all regions from the second operation into the first and delete the
  /// second operation.
  void mergeOperationsIntoFrom(Operation *op1, Operation *op2) {
    assert(op1 != op2 && "Cannot merge an op into itself");
    for (size_t i = 0, e = op1->getNumRegions(); i != e; ++i)
      mergeRegions(&op1->getRegion(i), &op2->getRegion(i));

    op2->erase();
    anythingChanged = true;
  }

  // This function merge two if operations by hoisting common conditions.
  sv::IfOp hoistIfOpConditions(sv::IfOp op1, sv::IfOp op2);

  // This function tries merging two if operations.
  sv::IfOp tryMergingIfOps(sv::IfOp ifOp, sv::IfOp prevIfOp) {
    assert(ifOp != prevIfOp && "Cannot merge an op into itself");

    // If conditions of op1 and op2 are equal, we can just merge them.
    if (ifOp.cond() == prevIfOp.cond()) {
      mergeOperationsIntoFrom(ifOp, prevIfOp);
      return ifOp;
    }

    // If convertIfToCase is not enabled, just return first if op.
    if (!convertIfToCase)
      return ifOp;

    return hoistIfOpConditions(ifOp, prevIfOp);
  }

  bool anythingChanged;
};
} // end anonymous namespace

void HWCleanupPass::runOnOperation() {
  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;
  runOnGraphRegion(getOperation().getBody());

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

/// Recursively process all of the regions in the specified op, dispatching to
/// graph or procedural processing as appropriate.
void HWCleanupPass::runOnRegionsInOp(Operation &op) {
  if (op.hasTrait<sv::ProceduralRegion>()) {
    for (auto &region : op.getRegions())
      runOnProceduralRegion(region);
  } else {
    for (auto &region : op.getRegions())
      runOnGraphRegion(region);
  }
}

/// Run simplifications on the specified graph region.
void HWCleanupPass::runOnGraphRegion(Region &region) {
  if (region.getBlocks().size() != 1)
    return;
  Block &body = region.front();

  // A set of operations in the current block which are mergable. Any
  // operation in this set is a candidate for another similar operation to
  // merge in to.
  DenseSet<Operation *, AlwaysLikeOpInfo> alwaysFFOpsSeen;
  llvm::SmallDenseMap<Attribute, Operation *, 4> ifdefOps;
  sv::InitialOp initialOpSeen;
  sv::AlwaysCombOp alwaysCombOpSeen;

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Merge alwaysff and always operations by hashing them to check to see if
    // we've already encountered one.  If so, merge them and reprocess the body.
    if (isa<sv::AlwaysOp, sv::AlwaysFFOp>(op)) {
      // Merge identical alwaysff's together and delete the old operation.
      auto itAndInserted = alwaysFFOpsSeen.insert(&op);
      if (itAndInserted.second)
        continue;
      auto *existingAlways = *itAndInserted.first;
      mergeOperationsIntoFrom(&op, existingAlways);

      *itAndInserted.first = &op;
      continue;
    }

    // Merge graph ifdefs anywhere in the module.
    if (auto ifdefOp = dyn_cast<sv::IfDefOp>(op)) {
      auto *&entry = ifdefOps[ifdefOp.condAttr()];
      if (entry)
        mergeOperationsIntoFrom(ifdefOp, entry);

      entry = ifdefOp;
      continue;
    }

    // Merge initial ops anywhere in the module.
    if (auto initialOp = dyn_cast<sv::InitialOp>(op)) {
      if (initialOpSeen)
        mergeOperationsIntoFrom(initialOp, initialOpSeen);
      initialOpSeen = initialOp;
      continue;
    }

    // Merge always_comb ops anywhere in the module.
    if (auto alwaysComb = dyn_cast<sv::AlwaysCombOp>(op)) {
      if (alwaysCombOpSeen)
        mergeOperationsIntoFrom(alwaysComb, alwaysCombOpSeen);
      alwaysCombOpSeen = alwaysComb;
      continue;
    }
  }

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Recursively process any regions in the op.
    if (op.getNumRegions() != 0)
      runOnRegionsInOp(op);
  }
}

/// Run simplifications on the specified procedural region.
void HWCleanupPass::runOnProceduralRegion(Region &region) {
  if (region.getBlocks().size() != 1)
    return;
  Block &body = region.front();

  Operation *lastSideEffectingOp = nullptr;
  for (auto it = body.begin(), end = body.end(); it != end; it++) {
    Operation &op = *it;
    // Merge procedural ifdefs with neighbors in the procedural region.
    if (auto ifdef = dyn_cast<sv::IfDefProceduralOp>(op)) {
      if (auto prevIfDef =
              dyn_cast_or_null<sv::IfDefProceduralOp>(lastSideEffectingOp)) {
        if (ifdef.cond() == prevIfDef.cond()) {
          // We know that there are no side effective operations between the
          // two, so merge the first one into this one.
          mergeOperationsIntoFrom(ifdef, prevIfDef);
        }
      }
    }

    // Merge 'if' operations.
    if (auto ifop = dyn_cast<sv::IfOp>(op)) {
      if (auto prevIf = dyn_cast_or_null<sv::IfOp>(lastSideEffectingOp)) {
        auto mergedIfOp = tryMergingIfOps(ifop, prevIf);
        // If a new if op is returned, we have to update lastSideEffectingOp and
        // the iterator because if ops are sunk.
        if (mergedIfOp != ifop) {
          lastSideEffectingOp = mergedIfOp;
          it = mergedIfOp->getIterator();
          continue;
        }
      }
    }

    // Keep track of the last side effecting operation we've seen.
    if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op))
      lastSideEffectingOp = &op;
  }

  // Convert if ops into casez.
  if (convertIfToCase)
    runIfOpsToCaseOnBlock(body);

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Recursively process any regions in the op.
    if (op.getNumRegions() != 0)
      runOnRegionsInOp(op);
  }
}

// This function merge two if operations when they have a common condition.
// For example,
// prevIfOp:
//   if a1 & a2 &... & an & c1 & c2 .. & cn {e1}
// ifOp:
//   if b1 & b2 &... & bn & c1 & c2 .. & cn {e2}
// ====>
//   if c1 & c2 .. & cn {
//     if a1 & a2 & ... & an {e1}
//     if b1 & b2 & ... & bn {e2}
//   }
sv::IfOp HWCleanupPass::hoistIfOpConditions(sv::IfOp ifOp, sv::IfOp prevIfOp) {
  // TOOD: Handle even when they have else blocks
  if (ifOp.hasElse() || prevIfOp.hasElse())
    return ifOp;

  ConditionPairInformation condPairInfo(ifOp.cond(), prevIfOp.cond());

  // If there is nothing in common, we cannot merge.
  if (condPairInfo.commonOperand.empty())
    return ifOp;

  // If both lhsOperands and rhsOperands are empty, it means the conditions
  // are actually equivalent.
  if (condPairInfo.lhsOperands.empty() && condPairInfo.rhsOperands.empty()) {
    mergeOperationsIntoFrom(ifOp, prevIfOp);
    return ifOp;
  }

  anythingChanged = true;

  OpBuilder builder(ifOp.getContext());
  auto i1Type = ifOp.cond().getType();

  auto generateCondValue =
      [&](Location loc, llvm::SmallSetVector<Value, 4> &conjunction) -> Value {
    if (conjunction.empty())
      return builder.create<hw::ConstantOp>(loc, i1Type, 1);

    return builder.createOrFold<comb::AndOp>(loc, i1Type,
                                             conjunction.takeVector());
  };

  auto merge = [&]() {
    // If rhsOperands is empty, it means we can move ifOp to the block of
    // prevIfOp.
    if (condPairInfo.rhsOperands.empty()) {
      // Move ifOp to the end of prevIfOp's then block.
      builder.setInsertionPointToEnd(prevIfOp.getThenBlock());
      auto newCond1 =
          generateCondValue(ifOp.cond().getLoc(), condPairInfo.lhsOperands);
      ifOp.setOperand(newCond1);
      // op1 might contain ops defined between op2 and op1, we have to move op2
      // to the position of op1 to ensure that the dominance doesn't break.
      prevIfOp->moveBefore(ifOp);
      ifOp->moveBefore(prevIfOp.getThenBlock(), prevIfOp.getThenBlock()->end());
      return prevIfOp;
    }

    // If lhsOperands is empty, it means we can move prevIfOp to the block of
    // ifOp.
    if (condPairInfo.lhsOperands.empty()) {
      // Move prevIfOp to the start of ifOp's then block.
      builder.setInsertionPointToStart(ifOp.getThenBlock());
      auto newCond2 =
          generateCondValue(prevIfOp.cond().getLoc(), condPairInfo.rhsOperands);
      prevIfOp.setOperand(newCond2);
      prevIfOp->moveAfter(ifOp.getThenBlock(), ifOp.getThenBlock()->begin());
      return ifOp;
    }

    // Ok, now we create a new if op to contain ifOp and prevIfOp.
    builder.setInsertionPoint(ifOp);

    // Create new conditions.
    auto newCond1 = generateCondValue(ifOp.getLoc(), condPairInfo.lhsOperands);
    auto newCond2 =
        generateCondValue(prevIfOp.getLoc(), condPairInfo.rhsOperands);
    auto cond = builder.createOrFold<comb::AndOp>(ifOp.getLoc(),
                                                  condPairInfo.commonOperand);
    auto newIf = builder.create<sv::IfOp>(prevIfOp.getLoc(), cond, [&]() {});

    prevIfOp->moveBefore(newIf.getThenBlock(), newIf.getThenBlock()->begin());
    ifOp->moveAfter(prevIfOp);
    ifOp.setOperand(newCond1);
    prevIfOp.setOperand(newCond2);
    return newIf;
  };

  auto returnedIfOp = merge();
  condPairInfo.cleanUpOps();
  return returnedIfOp;
}

// This function converts successive if ops into a case satatement by
// pattern matching.
void HWCleanupPass::runIfOpsToCaseOnBlock(Block &body) {
  // This indicates the value which might be used as a condition of casez.
  Value comparedValue;
  llvm::MapVector<APInt, SmallVector<sv::IfOp>> caseIntegerToIfOps;

  auto tryConstructingCaseZ = [&](Operation *op) {
    // If there is only one case, or `comparedValue` is not set yet, we don't
    // create casez.
    if (caseIntegerToIfOps.size() <= 1 || !comparedValue) {
      caseIntegerToIfOps.clear();
      comparedValue = nullptr;
      return;
    }

    auto caseIntegerAndIfOps = caseIntegerToIfOps.takeVector();
    // Renew the map since `takeVector` moves the underlying vector.
    caseIntegerToIfOps = llvm::MapVector<APInt, SmallVector<sv::IfOp>>();

    OpBuilder builder(op->getContext());
    builder.setInsertionPoint(op);
    // Create case op with `caseValueAndIfOps.size()` cases.
    auto casez = builder.create<sv::CaseOp>(
        comparedValue.getLoc(), CaseStmtType::CaseStmt, comparedValue,
        caseIntegerAndIfOps.size(),
        [&](size_t caseIdx) -> circt::sv::CasePattern {
          auto constant = caseIntegerAndIfOps[caseIdx].first;

          circt::sv::CasePattern thePattern =
              circt::sv::CasePattern(constant, op->getContext());
          return thePattern;
        });

    for (unsigned idx : llvm::seq(0ul, caseIntegerAndIfOps.size())) {
      auto *block = casez.getCases()[idx].block;
      auto &ifOps = caseIntegerAndIfOps[idx].second;
      // Append all blocks of registered ifOps.
      for (auto ifOp : ifOps) {
        block->getOperations().splice(block->end(),
                                      ifOp.getThenBlock()->getOperations());
        comb::ICmpOp icmp = cast<comb::ICmpOp>(ifOp.cond().getDefiningOp());
        ifOp.erase();
        eraseOpWithoutUse(icmp.rhs().getDefiningOp());
        eraseOpWithoutUse(icmp);
      }
    }

    comparedValue = nullptr;
  };

  Operation *op = nullptr;
  for (auto it = body.begin(), end = body.end(); it != end; it++) {
    op = &*it;

    // This function checks that it is possible to merge the op into the current
    // case. TODO: Consider to use mlir::PatternMatch utils.
    auto patternMatchIfOp = [&](Operation *op) {
      auto ifOp = dyn_cast<sv::IfOp>(op);
      if (!ifOp || ifOp.hasElse())
        return false;

      // Check that condition is ICmpOp.
      auto icmpOp = dyn_cast_or_null<comb::ICmpOp>(ifOp.cond().getDefiningOp());
      if (!icmpOp || icmpOp.predicate() != comb::ICmpPredicate::eq)
        return false;
      auto constant =
          dyn_cast_or_null<circt::hw::ConstantOp>(icmpOp.rhs().getDefiningOp());
      if (!constant)
        return false;

      // Check that the comaparison is consistent with the casez we are
      // creating.
      if (!comparedValue)
        comparedValue = icmpOp.lhs();
      else if (comparedValue != icmpOp.lhs())
        return false;

      // Register the if op.
      caseIntegerToIfOps[constant.getValue()].push_back(ifOp);
      return true;
    };

    bool success = patternMatchIfOp(op);
    if (!success && !mlir::MemoryEffectOpInterface::hasNoEffect(op))
      tryConstructingCaseZ(op);
  }

  // We also have to construct case in the end.
  if (op)
    tryConstructingCaseZ(op);
}

std::unique_ptr<Pass> circt::sv::createHWCleanupPass(bool convertIfToCase) {
  return std::make_unique<HWCleanupPass>(convertIfToCase);
}
