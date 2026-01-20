//===- CombineAssertLike.cpp - Combine Assertions --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Combines all assertions and assumptions into single operations
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"

#include "circt/Dialect/LTL/LTLTypes.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_COMBINEASSERTLIKEPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace verif;
using namespace hw;

namespace {
/// Combines all assertions and assumptions by conjoining their conditions into
/// a single assertion and assumption respectively, e.g.
/// ```
/// [...]
/// verif.assume %pred0 : i1
/// verif.assume %pred1 : i1
/// [...]
/// verif.assert %cond0 : i1
/// verif.assert %cond1 : i1
/// ```
/// will be converted into
/// ```
/// [...]
/// %pred = comb.and bin %pred0, %pred1 : i1
/// verif.assume %pred : i1
/// [...]
/// %cond = comb.and bin %cond0, %cond1 : i1
/// verif.assert %cond : i1
/// ```
/// The accumulation will happen on a per-block basis.
/// Most of the logic here will be to handle splitting things into blocks
struct CombineAssertLikePass
    : verif::impl::CombineAssertLikePassBase<CombineAssertLikePass> {
  void runOnOperation() override;

private:
  /// Maps that store all of the accumulated conditions per block
  /// for assertions and assumption found during our walk.
  /// This is then used to create a large conjunction of
  /// all of them in the end to be used as the only assert/assume.
  /// We don't expect there to be many asserts/assumes per module, therefore we
  /// can afford the small map.
  llvm::SmallDenseMap<Block *, llvm::SmallVector<Value>> assertConditions;
  llvm::SmallDenseMap<Block *, llvm::SmallVector<Value>> assumeConditions;

  /// Keep track of valid asserts/assumes that will later need to be erased
  llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> assertsToErase;
  llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> assumesToErase;

  /// Accumulates conditions of assertions and assumptions.
  /// Note that this only considers cases where the conditions are
  /// of type `i1`, and will not merge LTL properties.
  template <typename T>
  LogicalResult accumulateCondition(
      T &op, llvm::SmallDenseMap<Block *, llvm::SmallVector<Value>> &conds,
      llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> &opsToErase,
      OpBuilder &builder) {
    // Extract the condition and parent block the assertlike belongs to
    auto condition = op.getProperty();
    auto defop = op.getOperation();
    Block *parent = defop->getBlock();

    // Check that our condition isn't an ltl property, if so ignore
    if (!isa<ltl::PropertyType, ltl::SequenceType>(condition.getType())) {

      // Check for an optional enable signal
      auto enable = op.getEnable();
      // For i1 conditions, the enable signal can be folded
      // directly into the condition
      if (enable) {
        // Enable should always be reachable from the op, so it's safe to
        // accumulate right before the op
        builder.setInsertionPoint(defop);

        auto andop =
            comb::AndOp::create(builder, defop->getLoc(), condition, enable);

        // We then only need to store the conjunction not the condition
        conds[parent].push_back(andop);
      } else {
        // If no enable is present, we can directly accumulate the condition
        conds[parent].push_back(condition);
      }

      // We no longer need the existing assert/assume so request a removal
      opsToErase[parent].push_back(defop);
    }
    return success();
  }

  /// Combines all of the conditions in a given list into
  /// a single large conjunction
  template <typename AT>
  LogicalResult conjoinConditions(Operation *assertlike,
                                  llvm::SmallVectorImpl<Value> &conds,
                                  OpBuilder &builder) {

    // Check that we actually accumulated conditions, otherwise exit
    if (conds.empty())
      return success();

    // Set insertion
    builder.setInsertionPointAfter(assertlike);

    // Create a variadic conjunction
    auto andop = comb::AndOp::create(builder, assertlike->getLoc(), conds,
                                     /*two_state=*/false);

    // Create the final assert/assume using the accumulated condition
    AT::create(builder, andop.getLoc(), andop, /*enable=*/nullptr,
               /*label=*/nullptr);

    return success();
  }

  /// Conjoins all of the conditions in a given map on a per block basis
  /// then erases all replaced assertlike ops
  template <typename AT>
  LogicalResult conjoinAndErase(
      llvm::SmallDenseMap<Block *, llvm::SmallVector<Value>> &conds,
      llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> &opsToErase,
      OpBuilder &builder) {
    // For each block, conjoin the accumulated conditions
    for (auto [parent, ops] : opsToErase) {
      // Only conjoin assertions if there was more than one valid assert-like
      if (ops.size() > 1) {
        // Check that some assertions were found
        if (auto it = conds.find(parent); it != conds.end())
          // Conjoin the conditions into an assert and an assume respectively
          if (failed(conjoinConditions<AT>(ops.back(), conds[parent], builder)))
            return failure();

        // Erase the ops
        for (auto op : ops)
          op->erase();
      }
    }
    return success();
  }
};
} // namespace

void CombineAssertLikePass::runOnOperation() {
  hw::HWModuleOp hwModule = getOperation();
  OpBuilder builder(hwModule);

  // Walk over all assert-like ops and accumulate their conditions
  // then create a new comb.and op or two for assertions and
  // assumptions to conjoin their respective accumulated conditions.
  hwModule.walk([&](Operation *op) {
    // Only consider assertions and assumptions, not cover ops
    if (auto aop = dyn_cast<verif::AssertOp>(op))
      if (failed(accumulateCondition(aop, assertConditions, assertsToErase,
                                     builder)))
        return signalPassFailure();
    if (auto aop = dyn_cast<verif::AssumeOp>(op))
      if (failed(accumulateCondition(aop, assumeConditions, assumesToErase,
                                     builder)))
        return signalPassFailure();
  });

  // For each block, conjoin the accumulated conditions and get rid of the
  // replaced operations
  if (failed(conjoinAndErase<verif::AssertOp>(assertConditions, assertsToErase,
                                              builder)))
    return signalPassFailure();

  // Same for assumptions
  if (failed(conjoinAndErase<verif::AssumeOp>(assumeConditions, assumesToErase,
                                              builder)))
    return signalPassFailure();

  // Clear the data structures for pass reuse
  assertConditions.clear();
  assumeConditions.clear();
  assertsToErase.clear();
  assumesToErase.clear();
}
