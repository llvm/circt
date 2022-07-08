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
struct LowerBitindex : public LowerBitindexBase<LowerBitindex> {
  LowerBitindex() {
  }

  void runOnOperation() override {
    LLVM_DEBUG(
        llvm::dbgs() << "===- Running LowerBitindex Pass "
                        "------------------------------------------------===\n");

    auto circuit = getOperation();
    for (auto module : circuit.getBody()->getOps<FModuleOp>()) {
      auto *body = module.getBody();
      for (auto bitindex : body->getOps<BitindexOp>()) {
        ImplicitLocOpBuilder builder(bitindex.getLoc(), bitindex.getContext());
        auto bits = builder.create<BitsPrimOp>(bitindex.getResult().getType(), bitindex.input(), bitindex.index(), bitindex.index());
        bitindex.replaceAllUsesWith(bits.getResult());
        bitindex.erase();
        // auto builder = OpBuilder::atBlockBegin(body);
        // builder.create<BitsPrimOp>(bitindex.getType(), bitindex.index(), bitindex.index());
      }
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerBitindexPass() {
  return std::make_unique<LowerBitindex>();
}
