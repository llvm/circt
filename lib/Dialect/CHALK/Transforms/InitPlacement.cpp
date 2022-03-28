//===- InitPlacement.cpp - Initial Placement --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Initial Placement pass.  This pass processes CHALK
// cells, and produces a non-overlapping initial placement.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/CHALK/CHALKTypes.h"
#include "circt/Dialect/CHALK/Passes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Casting.h"

using namespace circt;
using namespace chalk;

using CellOpList = std::vector<Operation *>;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct InitPlacementPass : public InitPlacementBase<InitPlacementPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void InitPlacementPass::runOnOperation() {
  auto cell = getOperation();
  OpBuilder builder(&getContext());
  CellOpList ops;
  llvm::for_each(cell.getBody()->getOperations(), [&](Operation &op) {
    if (isa<RectangleOp>(op)) {
      ops.push_back(&op);
    }
  });

  size_t idx = 0;
  int64_t placeX = 0;
  for (auto *op : ops) {
    RectangleOp rectangle = dyn_cast<RectangleOp>(op);
    int64_t prevX = rectangle.width();
    IntegerAttr placeXAttr = builder.getI64IntegerAttr(placeX);
    if (idx != 0) {
      rectangle.xCoordAttr(placeXAttr);
    }
    placeX += prevX;
    idx++;
  }
}

std::unique_ptr<mlir::Pass> circt::chalk::createInitPlacementPass() {
  auto pass = std::make_unique<InitPlacementPass>();
  return pass;
}
