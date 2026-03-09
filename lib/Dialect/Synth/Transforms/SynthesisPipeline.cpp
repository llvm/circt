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
#include "circt/Conversion/CombToDatapath.h"
#include "circt/Conversion/CombToSynth.h"
#include "circt/Conversion/DatapathToComb.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/Datapath/DatapathPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
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
static void addOpName(SmallVectorImpl<std::string> &ops) {
  (ops.push_back(AllowedOpTy::getOperationName().str()), ...);
}
template <typename... OpToLowerTy>
static std::unique_ptr<Pass> createLowerVariadicPass(bool timingAware) {
  LowerVariadicOptions options;
  addOpName<OpToLowerTy...>(options.opNames);
  options.timingAware = timingAware;
  return createLowerVariadic(options);
}
void circt::synth::buildCombLoweringPipeline(
    OpPassManager &pm, const CombLoweringPipelineOptions &options) {
  {
    if (!options.disableDatapath) {
      // Lower variadic Mul into a binary op to enable datapath lowering.
      pm.addPass(createLowerVariadicPass<comb::MulOp>(options.timingAware));
      pm.addPass(createConvertCombToDatapath());
      pm.addPass(createSimpleCanonicalizerPass());
      if (options.synthesisStrategy == OptimizationStrategyTiming)
        pm.addPass(datapath::createDatapathReduceDelay());
      circt::ConvertDatapathToCombOptions datapathOptions;
      datapathOptions.timingAware = options.timingAware;
      pm.addPass(createConvertDatapathToComb(datapathOptions));
    }
    pm.addPass(createCSEPass());
    pm.addPass(createSimpleCanonicalizerPass());
    // Partially legalize Comb, then run CSE and canonicalization.
    circt::ConvertCombToSynthOptions convOptions;
    addOpName<comb::AndOp, comb::OrOp, comb::XorOp, comb::MuxOp, comb::ICmpOp,
              hw::ArrayGetOp, hw::ArraySliceOp, hw::ArrayCreateOp,
              hw::ArrayConcatOp, hw::AggregateConstantOp>(
        convOptions.additionalLegalOps);
    pm.addPass(circt::createConvertCombToSynth(convOptions));
  }
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());
  // Balance mux chains. For area oriented flow, we want to keep the mux chains
  // unless they are very deep.
  comb::BalanceMuxOptions balanceOptions{OptimizationStrategyTiming ? 16 : 64};
  pm.addPass(comb::createBalanceMux(balanceOptions));

  // Lower variadic ops before running full lowering to target IR.
  if (options.targetIR.getValue() == TargetIR::AIG) {
    // For AIG, lower variadic XoR since AIG cannot keep variadic
    // representation.
    pm.addPass(createLowerVariadicPass<comb::XorOp>(options.timingAware));
  } else if (options.targetIR.getValue() == TargetIR::MIG) {
    // For MIG, lower variadic And, Or, and Xor since MIG cannot keep variadic
    // representation.
    pm.addPass(createLowerVariadicPass<comb::AndOp, comb::OrOp, comb::XorOp>(
        options.timingAware));
  }

  pm.addPass(circt::hw::createHWAggregateToComb());
  circt::ConvertCombToSynthOptions convOptions;
  convOptions.targetIR = options.targetIR.getValue() == TargetIR::AIG
                             ? CombToSynthTargetIR::AIG
                             : CombToSynthTargetIR::MIG;
  pm.addPass(circt::createConvertCombToSynth(convOptions));
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void circt::synth::buildSynthOptimizationPipeline(
    OpPassManager &pm, const SynthOptimizationPipelineOptions &options) {
  // LowerWordToBits may not be scalable for large designs so conditionally
  // disable it. It's also worth considering keeping word-level representation
  // for faster synthesis.
  if (!options.disableWordToBits)
    pm.addPass(synth::createLowerWordToBits());
  pm.addPass(createCSEPass());
  // Run after LowerWordToBits for more precise timing-info & scalability.
  pm.addPass(createLowerVariadicPass(options.timingAware));
  pm.addPass(createStructuralHash());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.addPass(synth::createMaximumAndCover());
  pm.addPass(createLowerVariadicPass(options.timingAware));
  pm.addPass(createStructuralHash());

  // SOP balancing.
  if (!options.disableSOPBalancing) {
    SOPBalancingOptions sopOptions;
    // FIXME: The following is very small compared to the default value of ABC
    // (6/8) and mockturtle(4/25) due to inefficient implementation of
    // CutRewriter.
    sopOptions.maxCutInputSize = 4;
    sopOptions.maxCutsPerRoot = 4;
    pm.addPass(synth::createSOPBalancing(sopOptions));
    pm.addPass(createStructuralHash());
  }

  if (!options.abcCommands.empty()) {
    synth::ABCRunnerOptions abcOptions;
    abcOptions.abcPath = options.abcPath;
    abcOptions.abcCommands.assign(options.abcCommands.begin(),
                                  options.abcCommands.end());
    abcOptions.continueOnFailure = options.ignoreAbcFailures;
    pm.addPass(synth::createABCRunner(abcOptions));
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
  PassPipelineRegistration<SynthOptimizationPipelineOptions>(
      "synth-optimization-pipeline",
      "The default pipeline for AIG optimization pipeline",
      buildSynthOptimizationPipeline);
}
