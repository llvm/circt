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

void circt::synthesis::buildAIGLoweringPipeline(OpPassManager &pm) {
  {
    // Partially legalize Comb to AIG, run CSE and canonicalization.
    circt::ConvertCombToAIGOptions convOptions;
    partiallyLegalizeCombToAIG<comb::AndOp, comb::OrOp, comb::XorOp,
                               comb::MuxOp, comb::ICmpOp, hw::ArrayGetOp,
                               hw::ArraySliceOp, hw::ArrayCreateOp,
                               hw::ArrayConcatOp, hw::AggregateConstantOp>(
        convOptions.additionalLegalOps);
    pm.addPass(circt::createConvertCombToAIG(convOptions));
  }
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());

  pm.addPass(circt::hw::createHWAggregateToComb());
  pm.addPass(circt::createConvertCombToAIG());
  pm.addPass(createCSEPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void circt::synthesis::buildAIGOptimizationPipeline(
    OpPassManager &pm, const AIGOptimizationPipelineOptions &options) {

  pm.addPass(aig::createLowerVariadic());

  // TODO: LowerWordToBits is not scalable for large designs. Change to
  // conditionally enable the pass once the rest of the pipeline was able
  // to handle multibit operands properly.
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

void circt::synthesis::registerSynthesisPipeline() {
  PassPipelineRegistration<EmptyPipelineOptions>(
      "synthesis-aig-lowering-pipeline",
      "The default pipeline for until AIG lowering", buildAIGLoweringPipeline);
  PassPipelineRegistration<AIGOptimizationPipelineOptions>(
      "synthesis-aig-optimization-pipeline",
      "The default pipeline for AIG optimization pipeline",
      buildAIGOptimizationPipeline);
}
