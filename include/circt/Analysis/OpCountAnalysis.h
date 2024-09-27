//===- OpCountAnalysis.h - operation count analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving the quantity of different kinds of operation.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_OPCOUNT_ANALYSIS_H
#define CIRCT_ANALYSIS_OPCOUNT_ANALYSIS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class AnalysisManager;
} // namespace mlir
namespace circt {
namespace analysis {

struct OpCountAnalysis {
  OpCountAnalysis(Operation *moduleOp, mlir::AnalysisManager &am);

  size_t getOpCount(OperationName opName);
  DenseMap<size_t, size_t> getOperandCountMap(OperationName opName);

private:
  DenseMap<OperationName, size_t> opCounts;
  DenseMap<OperationName, DenseMap<size_t, size_t>> operandCounts;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_SCHEDULING_ANALYSIS_H
