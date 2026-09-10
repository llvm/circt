//===- DebugUnroll.cpp - fully unroll loops tagged hls.debug_unroll --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A debugging/testing pass that fully unrolls every affine.for carrying the
// unit attribute `hls.debug_unroll`. Lets a test pin a specific loop's unroll
// without relying on the auto-pipeline heuristic, so estimator behavior on the
// post-unroll IR can be exercised in isolation.
//
// Usage:
//   affine.for %i = 0 to 64 {hls.debug_unroll} { ... }
//   hls-est %s --debug-unroll
//
// Innermost-first so a tagged outer loop unrolls its (already unrolled) inner
// bodies cleanly. Only constant-trip loops are unrolled; a tagged dynamic-trip
// loop is left untouched with a debug note.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "circt/Dialect/Resource/Passes/DebugUnroll.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>

#define DEBUG_TYPE "debug-unroll"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Unit attribute name a loop carries to request full unrolling.
static constexpr llvm::StringLiteral kDebugUnrollAttr = "hls.debug_unroll";

struct DebugUnrollPass
    : public PassWrapper<DebugUnrollPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DebugUnrollPass)

  StringRef getArgument() const final { return "debug-unroll"; }
  StringRef getDescription() const final {
    return "Fully unroll every affine.for tagged with the `hls.debug_unroll` "
           "unit attribute (testing/debugging aid).";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect tagged loops first; unrolling mutates the IR, so we don't want to
    // walk and mutate simultaneously. Post-order (innermost-first) so unrolling
    // an outer tagged loop sees already-unrolled inner bodies.
    SmallVector<AffineForOp> tagged;
    func.walk([&](AffineForOp forOp) {
      if (forOp->hasAttr(kDebugUnrollAttr))
        tagged.push_back(forOp);
    });

    for (AffineForOp forOp : tagged) {
      // loopUnrollFull requires a statically known trip count.
      if (!getConstantTripCount(forOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "debug-unroll: skipping dynamic-trip loop @ "
                   << forOp.getLoc() << "\n");
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "debug-unroll: unrolling @ " << forOp.getLoc() << "\n");
      if (failed(loopUnrollFull(forOp)))
        LLVM_DEBUG(llvm::dbgs()
                   << "debug-unroll: loopUnrollFull failed @ "
                   << forOp.getLoc() << "\n");
    }
  }
};

} // namespace

namespace circt::hls_analysis {

std::unique_ptr<mlir::Pass> createDebugUnrollPass() {
  return std::make_unique<DebugUnrollPass>();
}

void registerDebugUnrollPass() { ::mlir::PassRegistration<DebugUnrollPass>(); }

} // namespace circt::hls_analysis
