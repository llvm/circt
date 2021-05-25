//===- SimpleCanonicalizer.cpp - A simplified Canonicalizer pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simplified canonicalizer pass that doesn't do CFG
// optimizations and other things that aren't helpful for many hardware IRs.
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace circt;

#define GEN_PASS_CLASSES
#include "circt/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// SimpleCanonicalizer
//===----------------------------------------------------------------------===//

namespace {
/// Canonicalize operations in nested regions.
struct SimpleCanonicalizer
    : public SimpleCanonicalizerBase<SimpleCanonicalizer> {
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *op : context->getRegisteredOperations())
      op->getCanonicalizationPatterns(owningPatterns, context);
    patterns = std::move(owningPatterns);
    return success();
  }
  void runOnOperation() override;

  mlir::FrozenRewritePatternSet patterns;
};
} // end anonymous namespace

void SimpleCanonicalizer::runOnOperation() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  // config.maxIterations = 1;
  (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(), patterns,
                                     config);
}

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> circt::createSimpleCanonicalizerPass() {
  return std::make_unique<SimpleCanonicalizer>();
}
