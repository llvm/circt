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

  DenseSet<Value> variables;

  for (auto bitindex : llvm::make_early_inc_range(getOperation().getOps<BitindexOp>())) {
    if (bitindex->getUses().empty()) {
      bitindex.erase();
      continue;
    }
    variables.insert(bitindex.input());
  }

  // make a new wire for each dest-bitindexed variable
  for (auto var : variables) {
    auto *defn = var.getDefiningOp();
    auto mod = getOperation();
    ImplicitLocOpBuilder builder(mod->getLoc(), mod.getContext());
    builder.setInsertionPointToStart(mod.getBody());
    if (defn) {
      builder.setLoc(defn->getLoc());
      builder.setInsertionPointAfter(defn);
    }
    if (auto i = var.getType().dyn_cast<IntType>()) {
      if (!i.hasWidth()) {
        signalPassFailure();
        return;
      }
      auto w = i.getWidth().getValue();
      auto wire = builder.create<WireOp>(FVectorType::get(UIntType::get(var.getContext(), 1), w));
      Value prev = builder.create<SubindexOp>(wire, 0);
      for (int i = 1; i < w; i++) {
        Value subidx = builder.create<SubindexOp>(wire, i);
        Value cat = builder.create<CatPrimOp>(subidx, prev);
        prev = cat;
      }

      for (auto &use : var.getUses()) {
        auto *op = use.getOwner();
        if (auto connect = dyn_cast<StrictConnectOp>(op)) {
          if (var == connect.dest()) {
            ImplicitLocOpBuilder builder(connect.getLoc(), connect);
            for (int i = 0; i < w; i++) {
              Value bitsOp = builder.create<BitsPrimOp>(connect.src(), i, i);
              Value subidxOp = builder.create<SubindexOp>(wire, i);
              builder.create<StrictConnectOp>(subidxOp, bitsOp);
            }
            connect.erase();
          }
        }
      }

      // connect here after replacing all other connects
      builder.create<StrictConnectOp>(var, prev);

      for (auto &use : var.getUses()) {
        auto *op = use.getOwner();
        if (auto bitindex = dyn_cast<BitindexOp>(op)) {
          ImplicitLocOpBuilder builder(bitindex.getLoc(), bitindex);
          Value subidx = builder.create<SubindexOp>(wire, bitindex.index());
          bitindex.replaceAllUsesWith(subidx);
          bitindex.erase();
        }
      }
    }
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerBitindexPass() {
  return std::make_unique<LowerBitIndexPass>();
}
