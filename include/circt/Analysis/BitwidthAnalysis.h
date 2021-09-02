//===- BitWidthAnalysis.h - Support for building backedges ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides flow-based forward bitwidth analysis.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_BITWIDTH_H
#define CIRCT_ANALYSIS_BITWIDTH_H

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/Optional.h"

#include <memory>

namespace circt {
class BitWidthAnalysisImpl;

/// Analyze the bitwidths of the operations within a top-level operation @p op.
/// @p saturationWidth delimits the width of a value which the analysis
/// determines may saturate.
/// As an example, index types incremented in a loop with dynamic bounds may
/// saturate..
class BitwidthAnalysis {
public:
  explicit BitwidthAnalysis(mlir::Operation *op, unsigned saturationWidth = 32);

  /// Returns the bit width estimate for the value @p v or empty if value was
  /// not present within the result set of the bitwidth analysis.
  llvm::Optional<unsigned> valueWidth(mlir::Value v) const;

private:
  std::shared_ptr<BitWidthAnalysisImpl> analysis;
};

} // namespace circt

#endif // CIRCT_ANALYSIS_BITWIDTH_H
