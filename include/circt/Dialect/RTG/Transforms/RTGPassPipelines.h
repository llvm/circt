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

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the randomization pipeline to the `OpPassManager`.
void buildRandomizationPipeline(mlir::OpPassManager &pm,
                                const RandomizationPipelineOptions &options);

/// Registers all pipelines for the `rtg` dialect.
void registerRTGPipelines();

} // namespace rtg
} // namespace circt

#endif // CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSPIPELINES_H
