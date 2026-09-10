//===- DebugUnroll.h - fully unroll loops tagged hls.debug_unroll ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the debug-unroll pass: fully unrolls every affine.for carrying the
// `hls.debug_unroll` unit attribute. A testing/debugging aid for exercising
// estimator behavior on post-unroll IR without the auto-pipeline heuristic.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RESOURCE_PASSES_DEBUGUNROLL_H
#define CIRCT_DIALECT_RESOURCE_PASSES_DEBUGUNROLL_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt::hls_analysis {

/// Create the debug-unroll pass. Fully unrolls every affine.for tagged with the
/// `hls.debug_unroll` unit attribute.
std::unique_ptr<mlir::Pass> createDebugUnrollPass();

/// Register the debug-unroll pass with the global pass registry so it is
/// reachable as `--debug-unroll` from the tool driver.
void registerDebugUnrollPass();

} // namespace circt::hls_analysis

#endif // CIRCT_DIALECT_RESOURCE_PASSES_DEBUGUNROLL_H