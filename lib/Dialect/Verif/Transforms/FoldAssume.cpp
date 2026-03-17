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
using namespace verif;
using namespace hw;

namespace {

/// Keep track of assumptions and assertions on a per block basis
llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> assertsToErase;
llvm::SmallDenseMap<Block *, llvm::SmallVector<Operation *>> assumesToErase;

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
  void runOnOperation() override;

private:
};
} // namespace

void FoldAssumePass::runOnOperation() {
  auto mlirModule = getOperation();
  OpBuilder builder(mlirModule);

  // Walks over both supported top-level ops (hw::HWModuleOp and
  // verif::FormalOp) and dispatches the internal matcher on assume ops
  mlirModule.walk([&](Operation *op) {
    if (!isa<hw::HWModuleOp, verif::FormalOp>(op))
      return;
  });
}
