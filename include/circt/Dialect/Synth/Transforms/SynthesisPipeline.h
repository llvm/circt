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

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// Pipeline Options
//===----------------------------------------------------------------------===//
namespace circt {
namespace synth {

/// Options for the aig lowering pipeline.
struct AIGLoweringPipelineOptions
    : public mlir::PassPipelineOptions<AIGLoweringPipelineOptions> {
  PassOptions::Option<bool> disableDatapath{
      *this, "disable-datapath",
      llvm::cl::desc("Disable datapath optimization passes"),
      llvm::cl::init(false)};
};

/// Options for the aig optimization pipeline.
struct AIGOptimizationPipelineOptions
    : public mlir::PassPipelineOptions<AIGOptimizationPipelineOptions> {
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
void buildAIGLoweringPipeline(mlir::OpPassManager &pm,
                              const AIGLoweringPipelineOptions &options);
void buildAIGOptimizationPipeline(
    mlir::OpPassManager &pm, const AIGOptimizationPipelineOptions &options);

/// Register the synthesis pipelines.
void registerSynthesisPipeline();

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHESISPIPELINE_H
