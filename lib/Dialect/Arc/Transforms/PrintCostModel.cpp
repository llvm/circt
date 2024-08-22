//===- DummyAnalysisTester.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a dummy pass to test the cost model results it doesn't do any thing.
// just walks over the ops to compute some statistics.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcCostModel.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "arc-print-cost-model"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_PRINTCOSTMODEL
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

namespace {
struct PrintCostModelPass
    : public arc::impl::PrintCostModelBase<PrintCostModelPass> {
  void runOnOperation() override;
};
} // namespace

void PrintCostModelPass::runOnOperation() {
  OperationCosts statVars;
  ArcCostModel arcCostModel;
  for (auto moduleOp : getOperation().getOps<hw::HWModuleOp>()) {
    moduleOp.walk([&](Operation *op) { statVars += arcCostModel.getCost(op); });
  }

  moduleCost = statVars.totalCost();
  packingCost = statVars.packingCost;
  shufflingCost = statVars.shufflingCost;
  vectoroizeOpsBodyCost = statVars.vectorizeOpsBodyCost;
  allVectorizeOpsCost = statVars.packingCost + statVars.shufflingCost +
                        statVars.vectorizeOpsBodyCost;
}

std::unique_ptr<Pass> arc::createPrintCostModelPass() {
  return std::make_unique<PrintCostModelPass>();
}
