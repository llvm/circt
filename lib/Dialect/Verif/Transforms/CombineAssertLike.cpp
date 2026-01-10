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
  // Maps that store all of the accumulated conditions per block
  // for assertions and assumption found during our walk.
  // This is then used to create a large conjunction of
  // all of them in the end to be used as the only assert/assume.
  llvm::DenseMap<Operation *, llvm::SmallVector<Value>> assertConditions;
  llvm::DenseMap<Operation *, llvm::SmallVector<Value>> assumeConditions;

  // Keep track of valid asserts/assumes and
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> opsToErase;

  // Pushes an entry found inside a block into its coresponding map
  // Creates a new vector for that block if none has been made before
  template <typename T, typename TOp>
  void pushBackMap(llvm::DenseMap<Operation *, llvm::SmallVector<T>> &map,
                   Operation *block, TOp op) {
    // Check if the map already initialized a slot for this parent
    // If not initialize the entry before we update the map
    if (auto it = map.find(block); it == map.end())
      map[block] = llvm::SmallVector<T>();

    // Update the map
    map[block].push_back(op);
  }

  // Accumulates conditions of assertions and assumptions.
  // Note that this only considers cases where the conditions are
  // of type `i1`, and will not merge LTL properties.
  template <typename T>
  LogicalResult accumulateCondition(
      T &op, llvm::DenseMap<Operation *, llvm::SmallVector<Value>> &conds,
      OpBuilder &builder) {
    // Extract the condition and parent block the assertlike belongs to
    auto condition = op.getProperty();
    auto parent = op.getOperation()->getParentOp();

    // Check that our condition isn't an ltl property, if so ignore
    if (!isa<ltl::PropertyType, ltl::SequenceType>(condition.getType())) {

      // Check for an optional enable signal
      auto enable = op.getEnable();
      if (enable) {
        // For i1 conditions, the enable signal can be folded
        // directly into the condition
        builder.setInsertionPointAfter(condition.getDefiningOp());
        auto andop = comb::AndOp::create(builder, condition.getLoc(), condition,
                                         op.getEnable());

        // We then only need to store the conjunction not the condition
        pushBackMap(conds, parent, andop);
      } else {
        // If no enble is present, we can directly accumulate the condition
        pushBackMap(conds, parent, condition);
      }

      // We no longer need the existing assert/assume so request a removal
      pushBackMap(opsToErase, parent, op.getOperation());
    }
    return success();
  }

  // Combines all of the conditions in a given list into
  // a single large conjunction
  template <typename AT>
  LogicalResult conjoinConditions(llvm::SmallVector<Value> &conds,
                                  OpBuilder &builder) {

    // Check that we actually accumulated conditions, otherwise exit
    if (conds.empty())
      return success();

    // Set insertion
    size_t last_cond = conds.size() - 1;
    builder.setInsertionPointAfter(conds[last_cond].getDefiningOp());

    // Creata a variadic conjunction
    auto andop = comb::AndOp::create(builder, conds[last_cond].getLoc(), conds,
                                     /*two_state=*/false);

    // Create the final assert/assume using the accumulated condition
    AT::create(builder, andop.getLoc(), andop, /*enable=*/nullptr,
               /*label=*/nullptr);

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
      if (failed(accumulateCondition(aop, assertConditions, builder)))
        signalPassFailure();
    if (auto aop = dyn_cast<verif::AssumeOp>(op))
      if (failed(accumulateCondition(aop, assumeConditions, builder)))
        signalPassFailure();
  });

  // For each block, conjoin the accumulated conditions
  for (auto [parent, ops] : opsToErase) {
    // Only conjoin assertions if there was more than one valid assert-like
    if (ops.size() > 1) {
      // Check that some assertions were found
      if (auto it = assertConditions.find(parent); it != assertConditions.end())
        // Conjoin the conditions into an assert and an assume respectively
        if (failed(conjoinConditions<verif::AssertOp>(assertConditions[parent],
                                                      builder)))
          signalPassFailure();

      // Same for assumptions
      if (auto it = assumeConditions.find(parent); it != assumeConditions.end())
        if (failed(conjoinConditions<verif::AssumeOp>(assumeConditions[parent],
                                                      builder)))
          signalPassFailure();

      // Erase the ops
      for (auto op : ops)
        op->erase();
    }
  }

  // Clear the data structures for pass reuse
  assertConditions.clear();
  assumeConditions.clear();
  opsToErase.clear();
}
