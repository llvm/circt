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
  {
    MemoryAllocationPassOptions passOptions;
    passOptions.useImmediates = options.memoriesAsImmediates;
    pm.addNestedPass<rtg::TestOp>(rtg::createMemoryAllocationPass());
  }
  pm.addPass(rtg::createLowerUniqueLabelsPass());
  pm.addNestedPass<rtg::TestOp>(rtg::createLinearScanRegisterAllocationPass());
  {
    auto &anyPm = pm.nestAny();
    anyPm.addPass(mlir::createCSEPass());
    anyPm.addPass(createSimpleCanonicalizerPass());
  }
  pm.addPass(rtg::createUniqueValidateOpsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void circt::rtg::registerRTGPipelines() {
  PassPipelineRegistration<RandomizationPipelineOptions>(
      "rtg-randomization-pipeline",
      "The default pipeline for randomizing RTG tests.",
      buildRandomizationPipeline);
}
