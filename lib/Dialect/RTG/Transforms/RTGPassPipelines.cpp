//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/Transforms/RTGPassPipelines.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void circt::rtg::buildRandomizationPipeline(
    OpPassManager &pm, const RandomizationPipelineOptions &options) {
  {
    ElaborationPassOptions passOptions;
    passOptions.seed = options.seed;
    pm.addPass(rtg::createElaborationPass(passOptions));
  }
  pm.addPass(rtg::createInlineSequencesPass());
  pm.addPass(createSymbolDCEPass());
  {
    auto &testPm = pm.nest<rtg::TestOp>();
    MemoryAllocationPassOptions passOptions;
    passOptions.useImmediates = options.memoriesAsImmediates;
    testPm.addPass(rtg::createMemoryAllocationPass(passOptions));
    // testPm.addPass(rtg::createLowerUniqueLabelsPass());
    // testPm.addPass(rtg::createLinearScanRegisterAllocationPass());
  }
  {
    auto &anyPm = pm.nestAny();
    anyPm.addPass(mlir::createCSEPass());
    anyPm.addPass(createSimpleCanonicalizerPass());
  }
  // pm.addPass(rtg::createUniqueValidateOpsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void circt::rtg::registerPipelines() {
  PassPipelineRegistration<RandomizationPipelineOptions>(
      "rtg-randomization-pipeline",
      "The default pipeline for randomizing RTG tests.",
      buildRandomizationPipeline);
}
