//===- DummyAnalysisTester.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a dummy pass to test the analysis passes it doesn't do any thing. It
// just walks over the ops to compute some statistics, you can add any
// statistics you need to compute.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcCostModel.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "arc-dummy-analysis-tester"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_DUMMYANALYSISTESTER
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

namespace {
struct DummyAnalysisTesterPass
    : public arc::impl::DummyAnalysisTesterBase<DummyAnalysisTesterPass> {
  void runOnOperation() override;

  // You can add any statistics you need to compute here.
  struct StatisticVars {
    size_t moduleCost{0};
    size_t packingCost{0};
    size_t shufflingCost{0};
    size_t vectoroizeOpsBodyCost{0};
    size_t allVectorizeOpsCost{0};
  };

  StatisticVars statVars;
};
} // namespace

void DummyAnalysisTesterPass::runOnOperation() {
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>()) {
    ArcCostModel arcCostModel;
    moduleOp.walk([&](Operation *op) {
      statVars.moduleCost += arcCostModel.getCost(op);
    });
    statVars.packingCost += arcCostModel.getPackingCost();
    statVars.shufflingCost += arcCostModel.getShufflingCost();
    statVars.vectoroizeOpsBodyCost += arcCostModel.getVectorizeOpsBodyCost();
    statVars.allVectorizeOpsCost += arcCostModel.getAllVectorizeOpsCost();
  }

  moduleCost = statVars.moduleCost;
  packingCost = statVars.packingCost;
  shufflingCost = statVars.shufflingCost;
  vectoroizeOpsBodyCost = statVars.vectoroizeOpsBodyCost;
  allVectorizeOpsCost = statVars.allVectorizeOpsCost;
}

std::unique_ptr<Pass> arc::createDummyAnalysisTesterPass() {
  return std::make_unique<DummyAnalysisTesterPass>();
}
