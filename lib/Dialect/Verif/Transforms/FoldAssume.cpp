//===- FoldAssume.cpp - Fold Assumptions --------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Folds all assumes into the enable signal of an assert
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
#define GEN_PASS_DEF_FOLDASSUMEPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;

namespace {

/// Implements assumes manually by combining all of them into a knowledge
/// caucus, that is then used as the antecedant of every assertion within the
/// same region. This assumes that `combine-assert-like` is run prior to running
/// this pass, as it will fail if there is more than one assertion and one
/// assumption per block. E.g.,
/// ```mlir
/// [...]
/// verif.assume %conjoinedPred : i1
/// [...]
/// verif.assert %cond : i1
/// ```
/// is converted to
/// ```mlir
/// [...]
/// [...]
/// verif.assert %cond if %conjoinedPred : i1
/// ```
/// This is a semantically preserving transformation in the formal back-ends
/// as assumes have the following underlying semantics:
///
///                          {a0, ..., ak} ∈ A
/// -------------------------------------------------------------------------
///  Γ; A ⊢ assume {a0, ... ak}; assert a -> assert ((a0 and .. and ak) => a)
///
/// Therefore, assertions become trivial when assumptions do no hold for a
/// concrete value, making them equivalent to an enable signal.
struct FoldAssumePass : verif::impl::FoldAssumePassBase<FoldAssumePass> {

  /// Only allow scheduling on verif::FormalOp and hw::HWModuleOp
  bool canScheduleOn(RegisteredOperationName opInfo) const override {
    return opInfo.getStringRef() == hw::HWModuleOp::getOperationName() ||
           opInfo.getStringRef() == verif::FormalOp::getOperationName();
  }

  void runOnOperation() override;

private:
  /// Keep track of assumptions and assertions on a per block basis
  llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> asserts;
  llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> assumes;

  /// Checks if thee current assertion/assumption is found multiple times in the
  /// block. If no other is found, save this assumption/assertion.
  LogicalResult findAssertlikes(
      Operation *defop,
      llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> &ops) {
    Block *blk = defop->getBlock();

    // Ignore assertlikes that contain ltl properties
    if (isa<ltl::PropertyType, ltl::SequenceType>(
            defop->getOperand(0).getType()))
      return success();

    // Make sure the block doesn't have any other entries
    if (auto it = ops.find(blk); it != ops.end())
      if ((it->getSecond().size() > 0)) {
        defop->emitError()
            << "Multiple " << defop->getName()
            << " found in the current block! Run `--combine-assert-like` "
            << "before running --fold-assume.";
        return failure();
      }

    // Store the assertlike op
    ops[blk].push_back(defop);
    return success();
  }

  /// Fold all assumes into their block's assert for all blocks given an mlir
  /// modules' opbuilder.
  LogicalResult foldAssumesIntoAsserts(OpBuilder &builder) {
    // boolean type
    auto i1 = builder.getI1Type();
    // For each assume, find the assert from the same block and augment it's
    // enable signal with the assume's condition.
    for (auto [blk, ops] : assumes) {
      // Check if the block had any assumptions
      if (ops.size() == 0)
        continue;

      // We should fail here if multiple assumptions were stored, we do
      // not accumulate the assumptions, as that is done by
      // verif::CombineAssertLikePass
      if (ops.size() > 1) {
        ops.back()->emitError(
            "Multiple assuptions found in the current block! Run "
            "`--combine-assert-like` before running --fold-assume.");
        return failure();
      }

      // Should be an assumption
      auto assumeOp = cast<verif::AssumeOp>(ops.front());
      Location loc = assumeOp.getLoc();
      Value cond = assumeOp.getProperty();

      // Add new ops after the assumption
      builder.setInsertionPoint(ops.front());

      // Look for a matching assertion, make that list is non-empty
      if (auto it = asserts.find(blk);
          it != asserts.end() && it->getSecond().size() > 0) {
        auto const &assertOps = it->getSecond();

        // We should fail here if multiple assertions were stored, as we
        // do not fold the assumption into every assertion in the block
        if (assertOps.size() > 1) {
          assertOps.back()->emitError(
              "Multiple assertions found in the current block! Run "
              "`--combine-assert-like` before running --fold-assume.");
          return failure();
        }

        // Should be an assertion now
        auto assertOp = cast<verif::AssertOp>(assertOps.front());
        builder.setInsertionPoint(assertOps.front());
        Location assertLoc = assertOp.getLoc();
        Value en = assertOp.getEnable();

        // Replace the enable signal with the assumption condition
        verif::AssertOp::create(
            builder, assertLoc, assertOp.getProperty(),
            // Conjoin enable signal with condition if needed
            en ? comb::AndOp::create(builder, assertLoc, en, cond) : cond,
            /*label=*/{});

        // Remove the old op
        assertOp->erase();
      } else {
        // If no matching assertion was found, make a trivial assertion
        // and set the enable signal with the assumption's condition, i.e.
        // create `assert 1 if %cond`
        auto tConst =
            hw::ConstantOp::create(builder, loc, IntegerAttr::get(i1, 1));
        verif::AssertOp::create(builder, loc, tConst, /*enable=*/cond,
                                /*label=*/{});
      }

      // Delete the assumption
      assumeOp->erase();
    }
    // Only succeed if all blocks were converted
    return success();
  }
};
} // namespace

void FoldAssumePass::runOnOperation() {
  Operation *module = getOperation();
  OpBuilder builder(module);

  // dispatches the internal matcher on assume ops
  // Skip any non-module-like op
  if (!isa<hw::HWModuleOp, verif::FormalOp>(module))
    return;

  // At this point we can just walk the internal ops
  WalkResult wr = module->walk([&](Operation *op) {
    if (isa<verif::AssumeOp, verif::AssertOp>(op))
      // Group all assertlikes per block
      if (failed(findAssertlikes(op,
                                 isa<verif::AssertOp>(op) ? asserts : assumes)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });

  // Check if the walk failed
  if (wr.wasInterrupted())
    return signalPassFailure();

  // Fold all assumes into their block's assert
  if (failed(foldAssumesIntoAsserts(builder)))
    return signalPassFailure();

  // Clear data structures in between modules
  asserts.clear();
  assumes.clear();
}
