//===- RtgToolOptions.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/rtgtool/RtgToolOptions.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
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
  if (options.getVerifyPasses())
    pm.enableVerifier(options.getVerifyPasses());

  if (options.getVerbosePassExecution())
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "rtgtool"));

  {
    // Initial canonicalization of the input IR.
    auto &anyPm = pm.nestAny();
    anyPm.addPass(mlir::createCSEPass());
    anyPm.addPass(createSimpleCanonicalizerPass());
  }

  if (options.getOutputFormat() == RtgToolOptions::OutputFormat::MLIR)
    return;

  {
    ElaborationPassOptions passOptions;
    passOptions.seed = options.getSeed();
    pm.addPass(rtg::createElaborationPass(passOptions));
  }
  pm.addPass(rtg::createInlineSequencesPass());
  pm.addPass(rtg::createLowerUniqueLabelsPass());
  pm.addNestedPass<rtg::TestOp>(rtg::createLinearScanRegisterAllocationPass());
  {
    auto &anyPm = pm.nestAny();
    anyPm.addPass(mlir::createCSEPass());
    anyPm.addPass(createSimpleCanonicalizerPass());
  }

  if (options.getOutputFormat() == RtgToolOptions::OutputFormat::ElaboratedMLIR)
    return;

  {
    EmitRTGISAAssemblyPassOptions passOptions;
    SmallVector<std::string> unsupported(options.getUnsupportedInstructions());
    passOptions.unsupportedInstructions = unsupported;
    passOptions.unsupportedInstructionsFile =
        options.getUnsupportedInstructionsFile();
    passOptions.splitOutput = options.getSplitOutput();
    passOptions.path = options.getOutputPath();
    pm.addPass(rtg::createEmitRTGISAAssemblyPass(passOptions));
  }
}
