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

OperationCosts ArcCostModel::getCost(Operation *op) {
  return computeOperationCost(op);
}

OperationCosts ArcCostModel::computeOperationCost(Operation *op) {
  if (auto it = opCostCache.find(op); it != opCostCache.end())
    return it->second;

  OperationCosts costs;

  if (isa<circt::comb::ConcatOp>(op))
    costs.normalCost = size_t(OperationCost::CONCATCOST);
  else if (isa<circt::comb::ExtractOp>(op))
    costs.normalCost = size_t(OperationCost::EXTRACTCOST);
  else if (auto vecOp = dyn_cast<arc::VectorizeOp>(op)) {
    // VectorizeOpCost = packingCost + shufflingCost + bodyCost
    OperationCosts inputVecCosts = getInputVectorsCost(vecOp);
    costs.packingCost += inputVecCosts.packingCost;
    costs.shufflingCost += inputVecCosts.shufflingCost;

    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block) {
          OperationCosts innerCosts = computeOperationCost(&innerOp);
          costs.vectorizeOpsBodyCost += innerCosts.totalCost();
        }
      }
    }
  } else if (auto callableOp = dyn_cast<CallOpInterface>(op)) {
    // Callable Op? then resolve!
    if (auto *calledOp = callableOp.resolveCallable())
      return opCostCache[callableOp] = computeOperationCost(calledOp);
  } else if (isa<func::FuncOp, arc::DefineOp, mlir::ModuleOp>(op)) {
    // Get the body cost
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto &innerOp : block)
          costs += computeOperationCost(&innerOp);
  } else
    costs.normalCost = size_t(OperationCost::NORMALCOST);

  return opCostCache[op] = costs;
}

OperationCosts ArcCostModel::getInputVectorsCost(VectorizeOp vecOp) {
  OperationCosts costs;
  for (auto inputVec : vecOp.getInputs()) {
    if (auto otherVecOp = inputVec[0].getDefiningOp<VectorizeOp>();
        all_of(inputVec.begin(), inputVec.end(), [&](auto element) {
          return element.template getDefiningOp<VectorizeOp>() == otherVecOp;
        })) {
      // This means that they came from the same vector or
      // VectorizeOp == null so they are all scalars

      // Check if they all scalars we multiply by the PACKCOST (SHL/R + OR)
      if (!otherVecOp)
        costs.packingCost += inputVec.size() * size_t(OperationCost::PACKCOST);
      else
        costs.shufflingCost += inputVec == otherVecOp.getResults()
                                   ? size_t(OperationCost::SAMEVECTORNOSHUFFLE)
                                   : getShufflingCost(inputVec, true);
    } else
      // inputVector consists of elements from different vectotrize ops and
      // may have scalars as well.
      costs.shufflingCost += getShufflingCost(inputVec);
  }

  return costs;
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
