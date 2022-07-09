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

  for (auto bitindex : llvm::make_early_inc_range(getOperation().getOps<BitindexOp>())) {
    if (bitindex->getUses().empty()) {
      bitindex->erase();
      continue;
    }
    auto drivers = make_filter_range(bitindex->getUsers(), [&](Operation *op) {
      if (auto connectOp = dyn_cast<ConnectOp>(op)) {

      }
    });
    ImplicitLocOpBuilder builder(bitindex.getLoc(), bitindex);
    Value replacement = builder.create<BitsPrimOp>(bitindex.getResult().getType(), bitindex.input(), bitindex.index(), bitindex.index());
    bitindex.replaceAllUsesWith(replacement);
    bitindex.erase();
  }
}

} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerBitindexPass() {
  return std::make_unique<LowerBitIndexPass>();
}
