//===- RtgToolOptions.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/rtgtool/RtgToolOptions.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// RTG Tool Pipelines
//===----------------------------------------------------------------------===//

void rtg::populateRandomizerPipeline(mlir::PassManager &pm,
                                     const RtgToolOptions &options) {
  pm.enableVerifier(options.getVerifyPasses());

  if (options.getVerbosePassExecution())
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "rtgtool"));

  pm.addPass(createSimpleCanonicalizerPass());
  if (options.getOutputFormat() != RtgToolOptions::OutputFormat::MLIR) {
    ElaborationOptions elabOptions;
    if (options.hasSeed())
      elabOptions.seed = options.getSeed();
    elabOptions.debugMode = options.getDebugMode();
    pm.addPass(createElaborationPass(elabOptions));
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createSimpleCanonicalizerPass());
  }
}
