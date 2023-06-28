//===- CSE.cpp - Common Sub-expression Elimination ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a simple common sub-expression elimination
// algorithm on operations within a region.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Transforms/Passes.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>

using namespace mlir;
using namespace circt;

namespace {

/// ----- Start of LLVM upstream changes.

/// TODO: Upstream following changes to mlir/IR/OperationSupport.h

//===----------------------------------------------------------------------===//
// Operation Equivalency
//===----------------------------------------------------------------------===//

struct OperationEquivalence {
  enum Flags {
    None = 0,

    // When provided, the location attached to the operation are ignored.
    IgnoreLocations = 1,

    LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ IgnoreLocations)
  };

  /// Compute a hash for the given operation.
  /// The `hashOperands` and `hashResults` callbacks are expected to return a
  /// unique hash_code for a given Value.
  static llvm::hash_code computeHash(
      Operation *op,
      function_ref<llvm::hash_code(Operation *)> hashOpInfo =
          [](Operation *op) {
            return llvm::hash_combine(
                op->getName(), op->getDiscardableAttrDictionary(),
                op->getResultTypes(), op->hashProperties());
          },
      function_ref<llvm::hash_code(Value)> hashOperands =
          [](Value v) { return hash_value(v); },
      function_ref<llvm::hash_code(Value)> hashResults =
          [](Value v) { return hash_value(v); },
      Flags flags = Flags::None);

  /// Helper that can be used with `computeHash` above to ignore operation
  /// operands/result mapping.
  static llvm::hash_code ignoreHashValue(Value) { return llvm::hash_code{}; }
  /// Helper that can be used with `computeHash` above to ignore operation
  /// operands/result mapping.
  static llvm::hash_code directHashValue(Value v) { return hash_value(v); }

  /// Compare two operations (including their regions) and return if they are
  /// equivalent.
  ///
  /// * `checkEquivalent` is a callback to check if two values are equivalent.
  ///   For two operations to be equivalent, their operands must be the same SSA
  ///   value or this callback must return `success`.
  /// * `markEquivalent` is a callback to inform the caller that the analysis
  ///   determined that two values are equivalent.
  ///
  /// Note: Additional information regarding value equivalence can be injected
  /// into the analysis via `checkEquivalent`. Typically, callers may want
  /// values that were determined to be equivalent as per `markEquivalent` to be
  /// reflected in `checkEquivalent`, unless `exactValueMatch` or a different
  /// equivalence relationship is desired.
  static bool isEquivalentTo(
      Operation *lhs, Operation *rhs,
      function_ref<LogicalResult(Operation *, Operation *)> checkOpProperties,
      function_ref<LogicalResult(Value, Value)> checkEquivalent,
      function_ref<void(Value, Value)> markEquivalent = nullptr,
      Flags flags = Flags::None);

  /// Compare two operations and return if they are equivalent.
  static bool isEquivalentTo(Operation *lhs, Operation *rhs, Flags flags);
  static bool isEquivalentTo(
      Operation *lhs, Operation *rhs,
      function_ref<LogicalResult(Operation *, Operation *)> checkOpProperties,
      Flags flags);

  /// Compare two regions (including their subregions) and return if they are
  /// equivalent. See also `isEquivalentTo` for details.
  static bool isRegionEquivalentTo(
      Region *lhs, Region *rhs,
      function_ref<LogicalResult(Operation *, Operation *)> checkOpProperties,
      function_ref<LogicalResult(Value, Value)> checkEquivalent,
      function_ref<void(Value, Value)> markEquivalent,
      OperationEquivalence::Flags flags);

  /// Helper that can be used with `isEquivalentTo` above.
  static LogicalResult simpleCheckOpProperties(Operation *lhs, Operation *rhs) {
    // Compare the operation properties.
    bool isNonequivalent = lhs->getName() != rhs->getName() ||
                           lhs->getDiscardableAttrDictionary() !=
                               rhs->getDiscardableAttrDictionary() ||
                           lhs->getNumRegions() != rhs->getNumRegions() ||
                           lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
                           lhs->getNumOperands() != rhs->getNumOperands() ||
                           lhs->getNumResults() != rhs->getNumResults() ||
                           lhs->hashProperties() != rhs->hashProperties();
    return failure(isNonequivalent);
  }

  /// Compare two regions and return if they are equivalent.
  static bool isRegionEquivalentTo(Region *lhs, Region *rhs,
                                   OperationEquivalence::Flags flags);

  /// Helper that can be used with `isEquivalentTo` above to consider ops
  /// equivalent even if their operands are not equivalent.
  static LogicalResult ignoreValueEquivalence(Value lhs, Value rhs) {
    return success();
  }
  /// Helper that can be used with `isEquivalentTo` above to consider ops
  /// equivalent only if their operands are the exact same SSA values.
  static LogicalResult exactValueMatch(Value lhs, Value rhs) {
    return success(lhs == rhs);
  }
};

/// TODO: Upstream following changes to mlir/IR/OperationSupport.cpp

llvm::hash_code OperationEquivalence::computeHash(
    Operation *op, function_ref<llvm::hash_code(Operation *)> hashOpInfo,
    function_ref<llvm::hash_code(Value)> hashOperands,
    function_ref<llvm::hash_code(Value)> hashResults, Flags flags) {
  // Hash operations using `hashOpInfo` based upon their:
  //   - Operation Name
  //   - Attributes
  //   - Result Types
  llvm::hash_code hash = hashOpInfo(op);

  //   - Operands
  ValueRange operands = op->getOperands();
  SmallVector<Value> operandStorage;
  if (op->hasTrait<mlir::OpTrait::IsCommutative>()) {
    operandStorage.append(operands.begin(), operands.end());
    llvm::sort(operandStorage, [](Value a, Value b) -> bool {
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    });
    operands = operandStorage;
  }
  for (Value operand : operands)
    hash = llvm::hash_combine(hash, hashOperands(operand));

  //   - Operands
  for (Value result : op->getResults())
    hash = llvm::hash_combine(hash, hashResults(result));
  return hash;
}

/*static*/ bool OperationEquivalence::isRegionEquivalentTo(
    Region *lhs, Region *rhs,
    function_ref<LogicalResult(Operation *, Operation *)> checkOpProperties,
    function_ref<LogicalResult(Value, Value)> checkEquivalent,
    function_ref<void(Value, Value)> markEquivalent,
    OperationEquivalence::Flags flags) {
  DenseMap<Block *, Block *> blocksMap;
  auto blocksEquivalent = [&](Block &lBlock, Block &rBlock) {
    // Check block arguments.
    if (lBlock.getNumArguments() != rBlock.getNumArguments())
      return false;

    // Map the two blocks.
    auto insertion = blocksMap.insert({&lBlock, &rBlock});
    if (insertion.first->getSecond() != &rBlock)
      return false;

    for (auto argPair :
         llvm::zip(lBlock.getArguments(), rBlock.getArguments())) {
      Value curArg = std::get<0>(argPair);
      Value otherArg = std::get<1>(argPair);
      if (curArg.getType() != otherArg.getType())
        return false;
      if (!(flags & OperationEquivalence::IgnoreLocations) &&
          curArg.getLoc() != otherArg.getLoc())
        return false;
      // Corresponding bbArgs are equivalent.
      if (markEquivalent)
        markEquivalent(curArg, otherArg);
    }

    auto opsEquivalent = [&](Operation &lOp, Operation &rOp) {
      // Check for op equality (recursively).
      if (!OperationEquivalence::isEquivalentTo(&lOp, &rOp, checkOpProperties,
                                                checkEquivalent, markEquivalent,
                                                flags))
        return false;
      // Check successor mapping.
      for (auto successorsPair :
           llvm::zip(lOp.getSuccessors(), rOp.getSuccessors())) {
        Block *curSuccessor = std::get<0>(successorsPair);
        Block *otherSuccessor = std::get<1>(successorsPair);
        auto insertion = blocksMap.insert({curSuccessor, otherSuccessor});
        if (insertion.first->getSecond() != otherSuccessor)
          return false;
      }
      return true;
    };
    return llvm::all_of_zip(lBlock, rBlock, opsEquivalent);
  };
  return llvm::all_of_zip(*lhs, *rhs, blocksEquivalent);
}

// Value equivalence cache to be used with `isRegionEquivalentTo` and
// `isEquivalentTo`.
struct ValueEquivalenceCache {
  DenseMap<Value, Value> equivalentValues;
  LogicalResult checkEquivalent(Value lhsValue, Value rhsValue) {
    return success(lhsValue == rhsValue ||
                   equivalentValues.lookup(lhsValue) == rhsValue);
  }
  void markEquivalent(Value lhsResult, Value rhsResult) {
    auto insertion = equivalentValues.insert({lhsResult, rhsResult});
    // Make sure that the value was not already marked equivalent to some other
    // value.
    (void)insertion;
    assert(insertion.first->second == rhsResult &&
           "inconsistent OperationEquivalence state");
  }
};

/*static*/ bool
OperationEquivalence::isRegionEquivalentTo(Region *lhs, Region *rhs,
                                           OperationEquivalence::Flags flags) {
  ValueEquivalenceCache cache;
  return isRegionEquivalentTo(
      lhs, rhs, OperationEquivalence::simpleCheckOpProperties,
      [&](Value lhsValue, Value rhsValue) -> LogicalResult {
        return cache.checkEquivalent(lhsValue, rhsValue);
      },
      [&](Value lhsResult, Value rhsResult) {
        cache.markEquivalent(lhsResult, rhsResult);
      },
      flags);
}

/*static*/ bool OperationEquivalence::isEquivalentTo(
    Operation *lhs, Operation *rhs,
    function_ref<LogicalResult(Operation *, Operation *)> checkOpProperties,
    function_ref<LogicalResult(Value, Value)> checkEquivalent,
    function_ref<void(Value, Value)> markEquivalent, Flags flags) {
  if (lhs == rhs)
    return true;

  // 1. Compare the operation properties.
  if (failed(checkOpProperties(lhs, rhs)))
    return false;
  if (!(flags & IgnoreLocations) && lhs->getLoc() != rhs->getLoc())
    return false;

  // 2. Compare operands.
  ValueRange lhsOperands = lhs->getOperands(), rhsOperands = rhs->getOperands();
  SmallVector<Value> lhsOperandStorage, rhsOperandStorage;
  if (lhs->hasTrait<mlir::OpTrait::IsCommutative>()) {
    auto sortValues = [](ValueRange values) {
      SmallVector<Value> sortedValues = llvm::to_vector(values);
      llvm::sort(sortedValues, [](Value a, Value b) {
        auto aArg = llvm::dyn_cast<BlockArgument>(a);
        auto bArg = llvm::dyn_cast<BlockArgument>(b);

        // Case 1. Both `a` and `b` are `BlockArgument`s.
        if (aArg && bArg) {
          if (aArg.getParentBlock() == bArg.getParentBlock())
            return aArg.getArgNumber() < bArg.getArgNumber();
          return aArg.getParentBlock() < bArg.getParentBlock();
        }

        // Case 2. One of then is a `BlockArgument` and other is not. Treat
        // `BlockArgument` as lesser.
        if (aArg && !bArg)
          return true;
        if (bArg && !aArg)
          return false;

        // Case 3. Both are values.
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      });
      return sortedValues;
    };
    lhsOperandStorage = sortValues(lhsOperands);
    lhsOperands = lhsOperandStorage;
    rhsOperandStorage = sortValues(rhsOperands);
    rhsOperands = rhsOperandStorage;
  }

  for (auto operandPair : llvm::zip(lhsOperands, rhsOperands)) {
    Value curArg = std::get<0>(operandPair);
    Value otherArg = std::get<1>(operandPair);
    if (curArg == otherArg)
      continue;
    if (curArg.getType() != otherArg.getType())
      return false;
    if (failed(checkEquivalent(curArg, otherArg)))
      return false;
  }

  // 3. Compare result types and mark results as equivalent.
  for (auto resultPair : llvm::zip(lhs->getResults(), rhs->getResults())) {
    Value curArg = std::get<0>(resultPair);
    Value otherArg = std::get<1>(resultPair);
    if (curArg.getType() != otherArg.getType())
      return false;
    if (markEquivalent)
      markEquivalent(curArg, otherArg);
  }

  // 4. Compare regions.
  for (auto regionPair : llvm::zip(lhs->getRegions(), rhs->getRegions()))
    if (!isRegionEquivalentTo(&std::get<0>(regionPair),
                              &std::get<1>(regionPair), checkOpProperties,
                              checkEquivalent, markEquivalent, flags))
      return false;

  return true;
}

/*static*/ bool OperationEquivalence::isEquivalentTo(
    Operation *lhs, Operation *rhs,
    function_ref<LogicalResult(Operation *, Operation *)> checkOpProperties,
    Flags flags) {
  ValueEquivalenceCache cache;
  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, checkOpProperties,
      [&](Value lhsValue, Value rhsValue) -> LogicalResult {
        return cache.checkEquivalent(lhsValue, rhsValue);
      },
      [&](Value lhsResult, Value rhsResult) {
        cache.markEquivalent(lhsResult, rhsResult);
      },
      flags);
}

/*static*/ bool OperationEquivalence::isEquivalentTo(Operation *lhs,
                                                     Operation *rhs,
                                                     Flags flags) {
  return OperationEquivalence::isEquivalentTo(
      lhs, rhs, OperationEquivalence::simpleCheckOpProperties, flags);
}

} // namespace

/// End of upstream changes to OperationEquivalence.

/// TODO: Upstream following changes to mlir/*/Transforms/CSE.{h, cpp}

namespace {
template <typename OperationInfo>
struct CSEImpl {
  CSEImpl(mlir::DominanceInfo *domInfo, Operation *rootOp)
      : domInfo(domInfo), rootOp(rootOp) {}
  /// Shared implementation of operation elimination and scoped map definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            OperationInfo, AllocatorTy>;

  /// Cache holding MemoryEffects information between two operations. The first
  /// operation is stored has the key. The second operation is stored inside a
  /// pair in the value. The pair also hold the MemoryEffects between those
  /// two operations. If the MemoryEffects is nullptr then we assume there is
  /// no operation with MemoryEffects::Write between the two operations.
  using MemEffectsCache =
      DenseMap<Operation *, std::pair<Operation *, MemoryEffects::Effect *>>;

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()) {}

    /// Scope for the known values.
    typename ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::const_iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed = false;
  };

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                  bool hasSSADominance);
  void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, Region &region);
  bool runCSE();
  unsigned getNumCSE() const { return numCSE; }
  unsigned getNumDCE() const { return numDCE; }

private:
  void replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                            Operation *existing, bool hasSSADominance);

  /// Check if there is side-effecting operations other than the given effect
  /// between the two operations.
  bool hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp);

  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;
  DominanceInfo *domInfo = nullptr;
  Operation *rootOp = nullptr;
  MemEffectsCache memEffectsCache;
  unsigned numCSE = 0, numDCE = 0;
};
} // namespace

template <typename OperationInfo>
void CSEImpl<OperationInfo>::replaceUsesAndDelete(ScopedMapTy &knownValues,
                                                  Operation *op,
                                                  Operation *existing,
                                                  bool hasSSADominance) {
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand in
  // an operation if it has not been visited yet.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    op->replaceAllUsesWith(existing);
    opsToErase.push_back(op);
  } else {
    // When the region does not have SSA dominance, we need to check if we
    // have visited a use before replacing any use.
    op->replaceUsesWithIf(existing->getResults(), [&](OpOperand &operand) {
      return !knownValues.count(operand.getOwner());
    });

    // There may be some remaining uses of the operation.
    if (op->use_empty())
      opsToErase.push_back(op);
  }

  // If the existing operation has an unknown location and the current
  // operation doesn't, then set the existing op's location to that of the
  // current op.
  if (isa<UnknownLoc>(existing->getLoc()) && !isa<UnknownLoc>(op->getLoc()))
    existing->setLoc(op->getLoc());

  ++numCSE;
}

template <typename OperationInfo>
bool CSEImpl<OperationInfo>::hasOtherSideEffectingOpInBetween(Operation *fromOp,
                                                              Operation *toOp) {
  assert(fromOp->getBlock() == toOp->getBlock());
  assert(
      isa<MemoryEffectOpInterface>(fromOp) &&
      cast<MemoryEffectOpInterface>(fromOp).hasEffect<MemoryEffects::Read>() &&
      isa<MemoryEffectOpInterface>(toOp) &&
      cast<MemoryEffectOpInterface>(toOp).hasEffect<MemoryEffects::Read>());
  Operation *nextOp = fromOp->getNextNode();
  auto result =
      memEffectsCache.try_emplace(fromOp, std::make_pair(fromOp, nullptr));
  if (result.second) {
    auto memEffectsCachePair = result.first->second;
    if (memEffectsCachePair.second == nullptr) {
      // No MemoryEffects::Write has been detected until the cached operation.
      // Continue looking from the cached operation to toOp.
      nextOp = memEffectsCachePair.first;
    } else {
      // MemoryEffects::Write has been detected before so there is no need to
      // check further.
      return true;
    }
  }
  while (nextOp && nextOp != toOp) {
    auto nextOpMemEffects = dyn_cast<MemoryEffectOpInterface>(nextOp);
    // TODO: Do we need to handle other effects generically?
    // If the operation does not implement the MemoryEffectOpInterface we
    // conservatively assumes it writes.
    if ((nextOpMemEffects &&
         nextOpMemEffects.hasEffect<MemoryEffects::Write>()) ||
        !nextOpMemEffects) {
      result.first->second =
          std::make_pair(nextOp, MemoryEffects::Write::get());
      return true;
    }
    nextOp = nextOp->getNextNode();
  }
  result.first->second = std::make_pair(toOp, nullptr);
  return false;
}

/// Attempt to eliminate a redundant operation.
template <typename OperationInfo>
LogicalResult
CSEImpl<OperationInfo>::simplifyOperation(ScopedMapTy &knownValues,
                                          Operation *op, bool hasSSADominance) {
  // Don't simplify terminator operations.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return failure();

  // If the operation is already trivially dead just add it to the erase list.
  if (isOpTriviallyDead(op)) {
    opsToErase.push_back(op);
    ++numDCE;
    return success();
  }

  // Don't simplify operations with regions that have multiple blocks.
  // TODO: We need additional tests to verify that we handle such IR correctly.
  if (!llvm::all_of(op->getRegions(), [](Region &r) {
        return r.getBlocks().empty() || llvm::hasSingleElement(r.getBlocks());
      }))
    return failure();

  // Some simple use case of operation with memory side-effect are dealt with
  // here. Operations with no side-effect are done after.
  if (!isMemoryEffectFree(op)) {
    auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
    // TODO: Only basic use case for operations with MemoryEffects::Read can be
    // eleminated now. More work needs to be done for more complicated patterns
    // and other side-effects.
    if (!memEffects || !memEffects.onlyHasEffect<MemoryEffects::Read>())
      return failure();

    // Look for an existing definition for the operation.
    if (auto *existing = knownValues.lookup(op)) {
      if (existing->getBlock() == op->getBlock() &&
          !hasOtherSideEffectingOpInBetween(existing, op)) {
        // The operation that can be deleted has been reach with no
        // side-effecting operations in between the existing operation and
        // this one so we can remove the duplicate.
        replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
        return success();
      }
    }
    knownValues.insert(op, op);
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
    ++numCSE;
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

template <typename OperationInfo>
void CSEImpl<OperationInfo>::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                                           bool hasSSADominance) {
  for (auto &op : *bb) {
    // Most operations don't have regions, so fast path that case.
    if (op.getNumRegions() != 0) {
      // If this operation is isolated above, we can't process nested regions
      // with the given 'knownValues' map. This would cause the insertion of
      // implicit captures in explicit capture only regions.
      if (op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
        ScopedMapTy nestedKnownValues;
        for (auto &region : op.getRegions())
          simplifyRegion(nestedKnownValues, region);
      } else {
        // Otherwise, process nested regions normally.
        for (auto &region : op.getRegions())
          simplifyRegion(knownValues, region);
      }
    }

    // If the operation is simplified, we don't process any held regions.
    if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance)))
      continue;
  }
  // Clear the MemoryEffects cache since its usage is by block only.
  memEffectsCache.clear();
}

template <typename OperationInfo>
void CSEImpl<OperationInfo>::simplifyRegion(ScopedMapTy &knownValues,
                                            Region &region) {
  // If the region is empty there is nothing to do.
  if (region.empty())
    return;

  bool hasSSADominance = domInfo->hasSSADominance(&region);

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    typename ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, &region.front(), hasSSADominance);
    return;
  }

  // If the region does not have dominanceInfo, then skip it.
  // TODO: Regions without SSA dominance should define a different
  // traversal order which is appropriate and can be used here.
  if (!hasSSADominance)
    return;

  // Note, deque is being used here because there was significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. If/when these performance issues are no
  // longer a problem we can change this to vector. For more information see
  // the llvm mailing list discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<std::unique_ptr<CFGStackNode>> stack;

  // Process the nodes of the dom tree for this region.
  stack.emplace_back(std::make_unique<CFGStackNode>(
      knownValues, domInfo->getRootNode(&region)));

  while (!stack.empty()) {
    auto &currentNode = stack.back();

    // Check to see if we need to process this node.
    if (!currentNode->processed) {
      currentNode->processed = true;
      simplifyBlock(knownValues, currentNode->node->getBlock(),
                    hasSSADominance);
    }

    // Otherwise, check to see if we need to process a child node.
    if (currentNode->childIterator != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(
          std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // Finally, if the node and all of its children have been processed
      // then we delete the node.
      stack.pop_back();
    }
  }
}
template <typename OperationInfo>
bool CSEImpl<OperationInfo>::runCSE() {
  /// A scoped hash table of defining operations within a region.
  ScopedMapTy knownValues;
  for (auto &region : rootOp->getRegions())
    simplifyRegion(knownValues, region);

  // If no operations were erased, then we mark all analyses as preserved.
  if (opsToErase.empty())
    return false;

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();
  return true;
}

/// ----- End of upstream changes to CSE.

/// ----- End of LLVM upstream changes.

namespace {

// DenseMapInfo that ignores namehints.
struct NamehintsInsensitiveOperationInfo
    : public llvm::DenseMapInfo<Operation *> {
  static bool isNotNamehint(mlir::NamedAttribute attr) {
    return attr.getName() != "sv.namehint";
  }
  static unsigned getHashValue(const Operation *opC) {
    // Hash operation information except for namehints.
    auto hashOpInfo = [](Operation *op) {
      // Hash the name, result types and properties.
      llvm::hash_code hash = llvm::hash_combine(
          op->getName(), op->getResultTypes(), op->hashProperties());
      mlir::DictionaryAttr attr = op->getDiscardableAttrDictionary();
      // If there is a namehint, construct a dictionary attr without a namehint.
      if (attr.contains("sv.namehint")) {
        SmallVector<mlir::NamedAttribute> attributes(
            llvm::make_filter_range(attr, isNotNamehint));
        attr = mlir::DictionaryAttr::get(op->getContext(), attributes);
      }

      // Hash attributes.
      hash = llvm::hash_combine(hash, attr);

      return hash;
    };
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC), hashOpInfo,
        OperationEquivalence::directHashValue,
        OperationEquivalence::ignoreHashValue);
  }

  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    auto checkOpProperties = [](Operation *lhs, Operation *rhs) {
      // Check basic properties other than discardable attributes.
      if (lhs->getName() != rhs->getName() ||
          lhs->getNumRegions() != rhs->getNumRegions() ||
          lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
          lhs->getNumOperands() != rhs->getNumOperands() ||
          lhs->getNumResults() != rhs->getNumResults() ||
          lhs->hashProperties() != rhs->hashProperties())
        return failure();

      // Ok, check equivalence of attributes expect for namehints.
      auto lDict = lhs->getDiscardableAttrDictionary();
      auto rDict = rhs->getDiscardableAttrDictionary();
      // We don't have to check if they are same.
      if (lDict == rDict)
        return success();

      // If neither has a namehint, they should be considered to be different.
      bool lHasNamehint = lDict.contains("sv.namehint");
      auto rHasNamehint = rDict.contains("sv.namehint");
      if (!lHasNamehint && !rHasNamehint)
        return failure();

      // Check their sizes.
      if (lDict.size() - static_cast<size_t>(lHasNamehint) !=
          rDict.size() - static_cast<size_t>(rHasNamehint))
        return failure();

      // Skip namehints and compare elements.
      for (auto [l, r] :
           llvm::zip(llvm::make_filter_range(lDict, isNotNamehint),
                     llvm::make_filter_range(rDict, isNotNamehint))) {
        if (l != r)
          return failure();
      }

      return success();
    };
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        checkOpProperties, OperationEquivalence::IgnoreLocations);
  }
};

/// Simple common sub-expression elimination.
struct NamehintsInsensitiveCSE
    : public NamehintsInsensitiveCSEBase<NamehintsInsensitiveCSE> {
  void runOnOperation() override;
};

} // namespace

void NamehintsInsensitiveCSE::runOnOperation() {
  auto &domInfo = getAnalysis<DominanceInfo>();
  CSEImpl<NamehintsInsensitiveOperationInfo> cseRunner(&domInfo,
                                                       getOperation());

  // Run cse, and preserve analyses if there was no change.
  if (!cseRunner.runCSE())
    return markAllAnalysesPreserved();
  numCSE += cseRunner.getNumCSE();
  numDCE += cseRunner.getNumDCE();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}

std::unique_ptr<Pass> circt::createNamehintsInsensitiveCSEPass() {
  return std::make_unique<NamehintsInsensitiveCSE>();
}
