//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSPIPELINES_H
#define CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace circt {
namespace rtg {

/// Options for the RTG randomization pipeline.
struct RandomizationPipelineOptions
    : public mlir::PassPipelineOptions<RandomizationPipelineOptions> {
  PassOptions::Option<unsigned> seed{*this, "seed",
                                     llvm::cl::desc("Seed for the RNG.")};
  PassOptions::Option<bool> memoriesAsImmediates{
      *this, "memories-as-immediates",
      llvm::cl::desc("Lower memories to immediates instead of labels."),
      llvm::cl::init(true)};
};

/// Options for the RTG emission pipeline.
struct EmissionPipelineOptions
    : public mlir::PassPipelineOptions<EmissionPipelineOptions> {
  PassOptions::Option<bool> splitOutput{
      *this, "split-output",
      llvm::cl::desc("If 'true' emits one file per 'rtg.test' in the IR. The "
                     "name of the file matches the test name and is placed in "
                     "'path'. Otherwise, path is interpreted as the full file "
                     "path including filename."),
      llvm::cl::init(false)};
  PassOptions::Option<std::string> path{
      *this, "path",
      llvm::cl::desc("The directory or file path in which the output files "
                     "should be created. If empty is is emitted to stderr (not "
                     "allowed if 'split-output' is set to 'true')")};
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the randomization pipeline to the `OpPassManager`.
void populateRandomizationPipeline(mlir::OpPassManager &pm,
                                   const RandomizationPipelineOptions &options);

/// Adds the emission pipeline to the `OpPassManager`.
void populateEmissionPipeline(mlir::OpPassManager &pm,
                              const EmissionPipelineOptions &options);

/// Registers all pipelines for the `rtg` dialect.
void registerPipelines();

} // namespace rtg
} // namespace circt

#endif // CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSPIPELINES_H
