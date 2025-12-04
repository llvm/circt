//===- EliminateWires.cpp - Eliminate useless wires -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EliminateWires pass.  This pass finds wires which,
// when relocated, become unnecessary and eliminates them.  This is done by
// creating a node between the write and readers.  Using a node allows the pass
// to preserve inner symbols and probes, letting  CSE remove useless nodes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <deque>

#define DEBUG_TYPE "firrtl-eliminate-wires"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_ELIMINATEWIRES
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct EliminateWiresPass
    : public circt::firrtl::impl::EliminateWiresBase<EliminateWiresPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void EliminateWiresPass::runOnOperation() {
  CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);

  auto module = getOperation();
  auto &dominance = getAnalysis<mlir::DominanceInfo>();

  std::deque<std::pair<WireOp, MatchingConnectOp>> worklist;

  for (auto wire : module.getOps<WireOp>()) {
    auto type = type_dyn_cast<FIRRTLBaseType>(wire.getResult().getType());
    if (!type || !type.isPassive()) {
      ++complexTypeWires;
      continue;
    }
    // This will fail if there are multiple connects (due to aggregate
    // decomposition or whens) or if there are connects in blocks (due to whens
    // or matches).  We are fine with this, we don't want to reason about those
    // cases as other passes will handle them.
    auto writer = getSingleConnectUserOf(wire.getResult());
    if (!writer) {
      ++complexWriteWires;
      continue;
    }
    bool safe = true;
    for (auto *user : wire->getUsers()) {
      if (!dominance.dominates(writer, user)) {
        ++notDomWires;
        safe = false;
        break;
      }
    }
    if (!safe)
      continue;
    worklist.emplace_back(wire, writer);
  }

  for (auto [wire, writer] : worklist) {
    mlir::ImplicitLocOpBuilder builder(wire->getLoc(), writer);
    auto node = NodeOp::create(builder, writer.getSrc(), wire.getName(),
                               wire.getNameKind(), wire.getAnnotations(),
                               wire.getInnerSymAttr(), wire.getForceable());
    wire.replaceAllUsesWith(node);
    wire.erase();
    writer.erase();
    ++erasedWires;
  }
}
