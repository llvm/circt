//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for transformation passes related to
// synthesis.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHPASSES_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHPASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <limits>

namespace circt {
namespace synth {
/// Optimization strategy.
/// Determines whether to prioritize area or timing.
enum OptimizationStrategy {
  OptimizationStrategyArea,  ///< Optimize for minimal area
  OptimizationStrategyTiming ///< Optimize for minimal critical path delay
};

#define GEN_PASS_DECL
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"

} // namespace synth

} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHPASSES_H
