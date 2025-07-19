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

#ifndef CIRCT_SYNTHESIS_TRANSFORMS_PASSES_H
#define CIRCT_SYNTHESIS_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <limits>

namespace circt {
namespace synthesis {
/// Optimization strategy.
/// Determines whether to prioritize area or timing.
enum OptimizationStrategy {
  OptimizationStrategyArea,  ///< Optimize for minimal area
  OptimizationStrategyTiming ///< Optimize for minimal critical path delay
};

#define GEN_PASS_DECL
#include "circt/Synthesis/Transforms/Passes.h.inc"
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Synthesis/Transforms/Passes.h.inc"

} // namespace synthesis

} // namespace circt

#endif // CIRCT_TRANSFORMS_PASSES_H
