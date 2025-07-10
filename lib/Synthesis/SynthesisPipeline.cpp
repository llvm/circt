//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the default synthesis pipeline from core dialect to AIG.
//
//===----------------------------------------------------------------------===//

#include "circt/Synthesis/SynthesisPipeline.h"
#include "circt/Conversion/CombToAIG.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace circt;
using namespace circt::synthesis;

//===----------------------------------------------------------------------===//
// Pipeline Implementation
//===----------------------------------------------------------------------===//

/// Helper function to populate additional legal ops for partial legalization.
template <typename... AllowedOpTy>
static void partiallyLegalizeCombToAIG(SmallVectorImpl<std::string> &ops) {
  (ops.push_back(AllowedOpTy::getOperationName().str()), ...);
}

/// Helper function to check if we should stop at a certain point.
static bool untilReached(SynthesisPipelineOptions::UntilPoint point,
                         SynthesisPipelineOptions::UntilPoint target) {
  return static_cast<int>(point) >= static_cast<int>(target);
}

void circt::synthesis::buildSynthesisPipeline(
    OpPassManager &pm, const SynthesisPipelineOptions &options) {
  auto pipeline = [&](OpPassManager &mpm) {
    {
      // Partially legalize Comb to AIG, run CSE and canonicalization.
      circt::ConvertCombToAIGOptions convOptions;
      partiallyLegalizeCombToAIG<comb::AndOp, comb::OrOp, comb::XorOp,
                                 comb::MuxOp, comb::ICmpOp, hw::ArrayGetOp,
                                 hw::ArraySliceOp, hw::ArrayCreateOp,
                                 hw::ArrayConcatOp, hw::AggregateConstantOp>(
          convOptions.additionalLegalOps);
      mpm.addPass(circt::createConvertCombToAIG(convOptions));
    }
    mpm.addPass(createCSEPass());
    mpm.addPass(createSimpleCanonicalizerPass());

    mpm.addPass(circt::hw::createHWAggregateToCombPass());
    mpm.addPass(circt::createConvertCombToAIG());
    mpm.addPass(createCSEPass());

    if (untilReached(SynthesisPipelineOptions::UntilAIGLowering,
                     options.runUntilAfter.getValue()))
      return;

    mpm.addPass(createSimpleCanonicalizerPass());
    mpm.addPass(createCSEPass());
    mpm.addPass(aig::createLowerVariadic());

    // TODO: LowerWordToBits is not scalable for large designs. Change to
    // conditionally enable the pass once the rest of the pipeline was able
    // to handle multibit operands properly.
    mpm.addPass(aig::createLowerWordToBits());
    mpm.addPass(createCSEPass());
    mpm.addPass(createSimpleCanonicalizerPass());

    if (!options.abcCommands.empty()) {
      aig::ABCRunnerOptions abcOptions;
      abcOptions.abcPath = options.abcPath;
      abcOptions.abcCommands.assign(options.abcCommands.begin(),
                                    options.abcCommands.end());
      abcOptions.continueOnFailure = options.ignoreAbcFailures;
      mpm.addPass(aig::createABCRunner(abcOptions));
    }

    // TODO: Add balancing, rewriting, FRAIG conversion, etc.
    assert(options.runUntilAfter.getValue() ==
           SynthesisPipelineOptions::UntilEnd);
  };

  if (options.topName.empty()) {
    pipeline(pm.nest<hw::HWModuleOp>());
  } else {
    pm.addPass(circt::createHierarchicalRunner(options.topName, pipeline));
  }

  // TODO: Add LUT mapping, etc.
}

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

void circt::synthesis::registerSynthesisPipeline() {
  PassPipelineRegistration<SynthesisPipelineOptions>(
      "synthesis-pipeline", "The default pipeline for synthesis pipeline",
      buildSynthesisPipeline);
}
