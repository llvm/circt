//===- PassiveWires.cpp - Make Wires Passive --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassiveWires pass.  This pass eliminated flips from
// wires with aggregate types.  Since flips only determine connect direction,
// they are unnecessary on wires and just get in the way.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-passive-wires"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_PASSIVEWIRES
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

static bool hasFlip(Type t) {
  if (auto type = type_dyn_cast<FIRRTLBaseType>(t))
    return !type.isPassive();
  return false;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct PassiveWiresPass
    : public circt::firrtl::impl::PassiveWiresBase<PassiveWiresPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void PassiveWiresPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n";);
  auto module = getOperation();

  // First, expand any connects to resolve flips.
  SmallVector<Operation *> worklist;
  module.walk([&](Operation *op) -> WalkResult {
    if (auto wire = dyn_cast<WireOp>(op)) {
      if (hasFlip(wire.getType(0)))
        worklist.push_back(wire);
      return WalkResult::advance();
    }
    if (!isa<ConnectOp, MatchingConnectOp>(op))
      return WalkResult::advance();
    // connect/matchingconnect
    if (!hasFlip(op->getOperand(0).getType()))
      return WalkResult::advance();

    mlir::ImplicitLocOpBuilder builder(op->getLoc(), op);
    // This will "blow out" a connect to passive pieces
    emitConnect(builder, op->getOperand(0), op->getOperand(1));
    op->erase();
    return WalkResult::advance();
  });

  // Second, remove flips from most things.
  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();
    auto r = op->getResult(0);
    if (!hasFlip(r.getType()))
      continue;
    for (auto users : r.getUsers())
      worklist.push_back(users);
    // In-place updates is safe as consumers don't care about flip.
    r.setType(type_cast<FIRRTLBaseType>(r.getType()).getPassiveType());
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createPassiveWiresPass() {
  return std::make_unique<PassiveWiresPass>();
}
