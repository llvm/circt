//===- LowerBitindex.cpp - Lower Bitindex  -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerBitindex pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <iostream>

#define DEBUG_TYPE "firrtl-lower-bitindex"

using namespace circt;
using namespace firrtl;

namespace {
struct LowerBitIndexPass : public LowerBitindexBase<LowerBitIndexPass> {
  void runOnOperation() override;
};

void LowerBitIndexPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===- Running LowerBitIndex Pass "
                      "------------------------------------------------===\n");

  SmallVector<BitindexOp> dests;
  SmallVector<BitindexOp> srcs;
  for (auto bitindex : llvm::make_early_inc_range(getOperation().getOps<BitindexOp>())) {
    for (auto &use : bitindex->getUses()) {
      auto *op = use.getOwner();
      if ((isa<ConnectOp>(op) && dyn_cast<ConnectOp>(op).dest() == bitindex) ||
          (isa<StrictConnectOp>(op) && dyn_cast<StrictConnectOp>(op).dest() == bitindex)) {
        dests.push_back(bitindex);
      } else if ((isa<ConnectOp>(op) && dyn_cast<ConnectOp>(op).src() == bitindex) ||
          (isa<StrictConnectOp>(op) && dyn_cast<StrictConnectOp>(op).src() == bitindex)) {
        auto bitsOp = OpBuilder(op).create<BitsPrimOp>(
          op->getLoc(),
          bitindex.getResult().getType(),
          bitindex.input(), bitindex.index(), bitindex.index()
        );
        use.set(bitsOp);
        srcs.push_back(bitindex);
      }
    }
  }

  for (auto op : dests) {
    auto *defn = op.input().getDefiningOp();
    auto mod = op->getParentOfType<FModuleOp>();
    ImplicitLocOpBuilder builder(mod->getLoc(), mod.getContext());
    builder.setInsertionPointToStart(mod.getBody());
    if (defn) {
      builder.setInsertionPointAfter(defn);
    }
    if (auto i = op.input().getType().dyn_cast<IntType>()) {
      if (!i.hasWidth()) {
        signalPassFailure();
        return;
      }
      builder.create<WireOp>(FVectorType::get(UIntType::get(op.getContext(), 1), i.getWidthOrSentinel()));
    }
  }

  // if (bitindex->getUses().empty()) {
  //   bitindex->erase();
  // }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerBitindexPass() {
  return std::make_unique<LowerBitIndexPass>();
}
