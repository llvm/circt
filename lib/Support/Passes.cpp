//===- Passes.cpp - Pass Utilities ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace circt;
using namespace mlir;

namespace circt {
/// A pass that runs verification on the operation. This pass always runs
/// verification, unlike the built-in pass manager verification which can be
/// skipped if the previous pass preserved all analyses.
struct VerifierPass : public PassWrapper<VerifierPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifierPass)

  StringRef getArgument() const override { return "verify-ir"; }
  StringRef getDescription() const override {
    return "Verify the IR at this point in the pipeline";
  }

  void runOnOperation() override {
    if (failed(mlir::verify(getOperation())))
      return signalPassFailure();
    // Mark all analyses as preserved since we didn't modify anything
    markAllAnalysesPreserved();
  }
};
} // namespace circt

std::unique_ptr<Pass> circt::createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  return mlir::createCanonicalizerPass(config);
}

std::unique_ptr<Pass> circt::createVerifierPass() {
  return std::make_unique<VerifierPass>();
}
