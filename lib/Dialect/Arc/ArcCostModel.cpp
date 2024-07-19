//===- ArcCostModel.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcCostModel.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>

using namespace llvm;
using namespace circt;
using namespace arc;
using namespace std;

size_t ArcCostModel::getCost(Operation *op) { return computeOperationCost(op); }

size_t ArcCostModel::computeOperationCost(Operation *op) {
  if (opCostCash.count(op))
    return opCostCash[op];
  if (isa<circt::comb::ConcatOp>(op))
    return opCostCash[op] = size_t(OperationCost::CONCATCOST);
  if (isa<circt::comb::ExtractOp>(op))
    return opCostCash[op] = size_t(OperationCost::EXTRACTCOST);
  // We have some other functions that need to be handled in a different way
  // arc::StateOp, arc::CallOp, mlir::func::CallOp and arc::VectorizeOp, each of
  // these functions have bodies so the cost of the op equals the cost of its
  // body.
  if (isa<arc::StateOp>(op) || isa<arc::CallOp>(op) ||
      isa<mlir::func::CallOp>(op)) {
    size_t totalCost = 0;
    const auto regions =
        dyn_cast<CallOpInterface>(op).resolveCallable()->getRegions();
    for (auto &region : regions)
      for (auto &block : region)
        for (auto &innerOp : block)
          totalCost += computeOperationCost(&innerOp);
    return opCostCash[op] = totalCost;
  }

  if (isa<arc::VectorizeOp>(op)) {
    size_t inputVecCost = getInputVectorsCost(dyn_cast<VectorizeOp>(op));
    size_t vecOpBodyCost = 0;
    auto regions = op->getRegions();
    for (auto &region : regions)
      for (auto &block : region)
        for (auto &innerOp : block)
          vecOpBodyCost += computeOperationCost(&innerOp);

    vectoroizeOpsBodyCost += vecOpBodyCost;
    allVectorizeOpsCost += inputVecCost + vecOpBodyCost;
    return opCostCash[op] = inputVecCost + vecOpBodyCost;
  }

  return opCostCash[op] = size_t(OperationCost::NORMALCOST);
}

size_t ArcCostModel::getInputVectorsCost(VectorizeOp vecOp) {
  // per VectorizeOp packing and shuffling costs
  size_t localPackCost = 0;
  size_t localShufflingCost = 0;
  for (auto inputVec : vecOp.getInputs()) {
    if (auto otherVecOp = inputVec[0].getDefiningOp<VectorizeOp>();
        all_of(inputVec.begin(), inputVec.end(), [&](auto element) {
          return element.template getDefiningOp<VectorizeOp>() == otherVecOp;
        })) {
      // This means that they came from the same vector or
      // VectorizeOp == null so they are all scalars

      // Check if they all scalars we multiply by the PACKCOST (SHL/R + OR)
      if (!otherVecOp)
        localPackCost += inputVec.size() * size_t(OperationCost::PACKCOST);
      else
        localShufflingCost += inputVec == otherVecOp.getResults()
                                  ? size_t(OperationCost::SAMEVECTORNOSHUFFLE)
                                  : getShufflingCost(inputVec, true);
    } else
      // inputVector consists of elements from different vectotrize ops and
      // may have scalars as well.
      localShufflingCost += getShufflingCost(inputVec);
  }
  packingCost += localPackCost;
  shufflingCost += localShufflingCost;
  return localShufflingCost + localPackCost;
}

size_t ArcCostModel::getShufflingCost(const ValueRange &inputVec, bool isSame) {
  size_t totalCost = 0;
  if (isSame) {
    auto vecOp = inputVec[0].getDefiningOp<VectorizeOp>();
    for (auto [elem, orig] : llvm::zip(inputVec, vecOp.getResults()))
      if (elem != orig)
        ++totalCost;

    return totalCost * size_t(OperationCost::SAMEVECTORSHUFFLECOST);
  }

  for (size_t i = 0; i < inputVec.size(); ++i) {
    auto otherVecOp = inputVec[i].getDefiningOp<VectorizeOp>();
    // If the element is not a result of a vector operation then it's a result
    // of a scalar operation, then it just needs to be packed into the vector.
    if (!otherVecOp)
      totalCost += size_t(OperationCost::PACKCOST);
    else {
      // If it's a result of a vector operation, then we have two cases:
      // (1) Its order in `inputVec` is the same as its order in the result of
      //     the defining op.
      // (2) the order is different.
      size_t idx = find(otherVecOp.getResults().begin(),
                        otherVecOp.getResults().end(), inputVec[i]) -
                   otherVecOp.getResults().begin();
      totalCost += i == idx ? size_t(OperationCost::DIFFERENTVECTORNOSHUFFLE)
                            : size_t(OperationCost::DIFFERENTVECTORSHUFFLECOST);
    }
  }
  return totalCost;
}
