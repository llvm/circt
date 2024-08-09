//===- ArcCostModel.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCCOSTMODEL_H
#define CIRCT_DIALECT_ARC_ARCCOSTMODEL_H

#include "circt/Dialect/Arc/ArcOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/AnalysisManager.h"

using namespace mlir;

namespace circt {
namespace arc {

// FIXME: May be refined and we have more accurate operation costs
enum class OperationCost : size_t {
  NOCOST,
  NORMALCOST,
  PACKCOST = 2,
  EXTRACTCOST = 3,
  CONCATCOST = 3,
  SAMEVECTORNOSHUFFLE = 0,
  SAMEVECTORSHUFFLECOST = 2,
  DIFFERENTVECTORNOSHUFFLE = 2,
  DIFFERENTVECTORSHUFFLECOST = 3
};

class ArcCostModel {
public:
  size_t getCost(Operation *op);
  size_t getPackingCost() const { return packingCost; }
  // This is a public interface for other passes to call
  size_t getShufflingCost() const { return shufflingCost; }
  size_t getVectorizeOpsBodyCost() const { return vectoroizeOpsBodyCost; }
  size_t getAllVectorizeOpsCost() const { return allVectorizeOpsCost; }

private:
  size_t computeOperationCost(Operation *op);

  // gets the cost to pack the vectors we have some cases we need to consider:
  // 1: the input is scalar so we can give it a cost of 1
  // 2: the input is a result of another vector but with no shuffling so the
  //    is 0
  // 3: the input is a result of another vector but with some shuffling so
  //    the cost is the (number of out of order elements) * 2
  // 4: the input is a mix of some vectors:
  //    a) same order we multiply by 2
  //    b) shuffling we multiply by 3
  size_t getInputVectorsCost(VectorizeOp vecOp);
  size_t getShufflingCost(const ValueRange &inputVec, bool isSame = false);
  DenseMap<Operation *, size_t> opCostCash;
  size_t packingCost{0};
  size_t shufflingCost{0};
  size_t vectoroizeOpsBodyCost{0};
  size_t allVectorizeOpsCost{0};
};

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCCOSTMODEL_H
