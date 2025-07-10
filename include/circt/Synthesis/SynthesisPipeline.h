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

#ifndef LIB_SYNTHESIS_SYNTHESISPIPELINE_H
#define LIB_SYNTHESIS_SYNTHESISPIPELINE_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include <string>
#include <vector>

namespace mlir {
class PassManager;
class OpPassManager;
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pipeline Options
//===----------------------------------------------------------------------===//
namespace circt {
namespace synthesis {

/// Options for the synthesis pipeline.
struct SynthesisPipelineOptions
    : public mlir::PassPipelineOptions<SynthesisPipelineOptions> {

  /// Enumeration for pipeline stopping points.
  enum UntilPoint { UntilAIGLowering = 0, UntilEnd = 1 };

  PassOptions::Option<std::string> topName{
      *this, "top", llvm::cl::desc("Top module name"), llvm::cl::init("")};

  PassOptions::ListOption<std::string> abcCommands{
      *this, "abc-commands", llvm::cl::desc("ABC passes to run")};

  PassOptions::Option<std::string> abcPath{
      *this, "abc-path", llvm::cl::desc("Path to ABC"), llvm::cl::init("abc")};

  PassOptions::Option<bool> ignoreAbcFailures{
      *this, "ignore-abc-failures",
      llvm::cl::desc("Continue on ABC failure instead of aborting"),
      llvm::cl::init(false)};

  PassOptions::Option<UntilPoint> runUntilAfter{
      *this, "until-after",
      llvm::cl::desc("Stop pipeline after a specified point"),
      llvm::cl::init(UntilEnd),
      llvm::cl::values(
          clEnumValN(UntilAIGLowering, "aig-lowering", "Lowering of AIG"),
          clEnumValN(UntilEnd, "all", "Run entire pipeline (default)"))};
};
//===----------------------------------------------------------------------===//
// Pipeline Functions
//===----------------------------------------------------------------------===//

/// Populate the synthesis pipeline with the given options.
void buildSynthesisPipeline(mlir::OpPassManager &pm,
                            const SynthesisPipelineOptions &options);

/// Register the synthesis pipeline.
void registerSynthesisPipeline();
} // namespace synthesis

} // namespace circt
#endif // LIB_SYNTHESIS_SYNTHESISPIPELINE_H
