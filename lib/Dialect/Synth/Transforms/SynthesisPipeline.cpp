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

#include "circt/Dialect/Synth/Transforms/SynthesisPipeline.h"
#include "circt/Conversion/CombToAIG.h"
#include "circt/Conversion/CombToDatapath.h"
#include "circt/Conversion/DatapathToComb.h"
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
using namespace circt::synth;

//===----------------------------------------------------------------------===//
// Pipeline Implementation
//===----------------------------------------------------------------------===//

/// Helper function to populate additional legal ops for partial legalization.
template <typename... AllowedOpTy>
static void partiallyLegalizeCombToSynth(SmallVectorImpl<std::string> &ops) {
  (ops.push_back(AllowedOpTy::getOperationName().str()), ...);
}

void circt::synth::buildCombLoweringPipeline(
    OpPassManager &pm, const CombLoweringPipelineOptions &options) {
  {
    if (!options.disableDatapath) {
      pm.addPass(createConvertCombToDatapath());
      pm.addPass(createSimpleCanonicalizerPass());
      circt::ConvertDatapathToCombOptions datapathOptions;
      datapathOptions.timingAware = options.timingAware;
      pm.addPass(createConvertDatapathToComb(datapathOptions));
      pm.addPass(createSimpleCanonicalizerPass());
    }
    // Partially legalize Comb, then run CSE and canonicalization.
    circt::ConvertCombToAIGOptions convOptions;
    partiallyLegalizeCombToSynth<comb::AndOp, comb::OrOp, comb::XorOp,
                                 comb::MuxOp, comb::ICmpOp, hw::ArrayGetOp,
                                 hw::ArraySliceOp, hw::ArrayCreateOp,
                                 hw::ArrayConcatOp, hw::AggregateConstantOp>(
        convOptions.additionalLegalOps);
    pm.addPass(circt::createConvertCombToAIG(convOptions));
  }
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());

  pm.addPass(circt::hw::createHWAggregateToComb());
  circt::ConvertCombToAIGOptions convOptions;
  convOptions.targetIR = options.targetIR.getValue() == TargetIR::AIG
                             ? CombToAIGTargetIR::AIG
                             : CombToAIGTargetIR::MIG;
  pm.addPass(circt::createConvertCombToAIG(convOptions));
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void circt::synth::buildAIGOptimizationPipeline(
    OpPassManager &pm, const AIGOptimizationPipelineOptions &options) {

  pm.addPass(aig::createLowerVariadic());

  // LowerWordToBits may not be scalable for large designs so conditionally
  // disable it. It's also worth considering keeping word-level representation
  // for faster synthesis.
  if (!options.disableWordToBits)
    pm.addPass(aig::createLowerWordToBits());
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());

  if (!options.abcCommands.empty()) {
    aig::ABCRunnerOptions abcOptions;
    abcOptions.abcPath = options.abcPath;
    abcOptions.abcCommands.assign(options.abcCommands.begin(),
                                  options.abcCommands.end());
    abcOptions.continueOnFailure = options.ignoreAbcFailures;
    pm.addPass(aig::createABCRunner(abcOptions));
  }
  // TODO: Add balancing, rewriting, FRAIG conversion, etc.
}

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

void circt::synth::registerSynthesisPipeline() {
  PassPipelineRegistration<CombLoweringPipelineOptions>(
      "synth-comb-lowering-pipeline",
      "The default pipeline for until Comb lowering",
      buildCombLoweringPipeline);
  PassPipelineRegistration<AIGOptimizationPipelineOptions>(
      "synth-aig-optimization-pipeline",
      "The default pipeline for AIG optimization pipeline",
      buildAIGOptimizationPipeline);
}
