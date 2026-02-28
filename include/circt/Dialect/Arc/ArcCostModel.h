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

struct OperationCosts {
  size_t normalCost{0};
  size_t packingCost{0};
  size_t shufflingCost{0};
  size_t vectorizeOpsBodyCost{0};
  size_t totalCost() const {
    return normalCost + packingCost + shufflingCost + vectorizeOpsBodyCost;
  }
  OperationCosts &operator+=(const OperationCosts &other) {
    this->normalCost += other.normalCost;
    this->packingCost += other.packingCost;
    this->shufflingCost += other.shufflingCost;
    this->vectorizeOpsBodyCost += other.vectorizeOpsBodyCost;
    return *this;
  }
};

class ArcCostModel {
public:
  OperationCosts getCost(Operation *op);

private:
  OperationCosts computeOperationCost(Operation *op);

  // gets the cost to pack the vectors we have some cases we need to consider:
  // 1: the input is scalar so we can give it a cost of 1
  // 2: the input is a result of another vector but with no shuffling so the
  //    is 0
  // 3: the input is a result of another vector but with some shuffling so
  //    the cost is the (number of out of order elements) * 2
  // 4: the input is a mix of some vectors:
  //    a) same order we multiply by 2
  //    b) shuffling we multiply by 3
  OperationCosts getInputVectorsCost(VectorizeOp vecOp);
  size_t getShufflingCost(const ValueRange &inputVec, bool isSame = false);
  DenseMap<Operation *, OperationCosts> opCostCache;
};

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCCOSTMODEL_H
