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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
using namespace circt;

#define GEN_PASS_CLASSES
#include "circt/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// SimpleCanonicalizer
//===----------------------------------------------------------------------===//

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> circt::createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return mlir::createCanonicalizerPass(config);
}
