//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the default synthesis pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHESISPIPELINE_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHESISPIPELINE_H

#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// Pipeline Options
//===----------------------------------------------------------------------===//
namespace circt {
namespace synth {

enum TargetIR {
  // Lower to And-Inverter Graph
  AIG,
  // Lower to Majority-Inverter Graph
  MIG
};

/// Options for the aig lowering pipeline.
struct CombLoweringPipelineOptions
    : public mlir::PassPipelineOptions<CombLoweringPipelineOptions> {
  PassOptions::Option<bool> disableDatapath{
      *this, "disable-datapath",
      llvm::cl::desc("Disable datapath optimization passes"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> timingAware{
      *this, "timing-aware",
      llvm::cl::desc("Lower operators in a timing-aware fashion"),
      llvm::cl::init(false)};
  PassOptions::Option<TargetIR> targetIR{
      *this, "lowering-target", llvm::cl::desc("Target IR to lower to"),
      llvm::cl::init(TargetIR::AIG)};
  PassOptions::Option<OptimizationStrategy> synthesisStrategy{
      *this, "synthesis-strategy", llvm::cl::desc("Synthesis strategy to use"),
      llvm::cl::values(
          clEnumValN(OptimizationStrategyArea, "area", "Optimize for area"),
          clEnumValN(OptimizationStrategyTiming, "timing",
                     "Optimize for timing")),
      llvm::cl::init(OptimizationStrategyTiming)};
};

/// Options for the synth optimization pipeline.
struct SynthOptimizationPipelineOptions
    : public mlir::PassPipelineOptions<SynthOptimizationPipelineOptions> {
  PassOptions::ListOption<std::string> abcCommands{
      *this, "abc-commands", llvm::cl::desc("ABC passes to run")};

  PassOptions::Option<std::string> abcPath{
      *this, "abc-path", llvm::cl::desc("Path to ABC"), llvm::cl::init("abc")};

  PassOptions::Option<bool> ignoreAbcFailures{
      *this, "ignore-abc-failures",
      llvm::cl::desc("Continue on ABC failure instead of aborting"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> disableWordToBits{
      *this, "disable-word-to-bits",
      llvm::cl::desc("Disable LowerWordToBits pass"), llvm::cl::init(false)};
};

//===----------------------------------------------------------------------===//
// Pipeline Functions
//===----------------------------------------------------------------------===//

/// Populate the synthesis pipelines.
void buildCombLoweringPipeline(mlir::OpPassManager &pm,
                               const CombLoweringPipelineOptions &options);
void buildSynthOptimizationPipeline(
    mlir::OpPassManager &pm, const SynthOptimizationPipelineOptions &options);

/// Register the synthesis pipelines.
void registerSynthesisPipeline();

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHESISPIPELINE_H
