//===- StrictWires.cpp - Make Wires Strict ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the StrictWires pass.  This pass converts passive wires
// to strict wires.  Strict wires have read and write ports.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-strict-wires"

using namespace circt;
using namespace firrtl;

template<typename RetTy>
static RetTy cloneWireTo(WireOp wire);

static void updateWireUses(Value toReplace, Value readSide, Value writeSide);

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct StrictWiresPass : public StrictWiresBase<StrictWiresPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void StrictWiresPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n";);
  auto module = getOperation();
  mlir::OpBuilder builder(module);

  module.walk([&](Operation *op) -> WalkResult {
    if (auto wire = dyn_cast<WireOp>(op)) {
      auto newWire = cloneWireTo<StrictWireOp>(wire);
      updateWireUses(wire.getResult(), newWire.getRead(), newWire.getWrite());
    }
    return WalkResult::advance();
  });
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createStrictWiresPass() {
  return std::make_unique<StrictWiresPass>();
}
