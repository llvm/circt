//===- DebugAnalysis.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_DEBUGANALYSIS_H
#define CIRCT_ANALYSIS_DEBUGANALYSIS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
class Operation;
class OpOperand;
} // namespace mlir

namespace circt {

/// Identify operations and values that are only used for debug info.
struct DebugAnalysis {
  DebugAnalysis(Operation *op);

  DenseSet<Operation *> debugOps;
  DenseSet<Value> debugValues;
  DenseSet<OpOperand *> debugOperands;
};

} // namespace circt

#endif // CIRCT_ANALYSIS_DEBUGANALYSIS_H
