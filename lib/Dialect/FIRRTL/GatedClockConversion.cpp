//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GatedClockConversion utility class.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/GatedClockConversion.h"
#include "circt/Dialect/FIRRTL/FIRRTLEnums.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "firrtl-gated-clock-conversion"

using namespace circt;
using namespace firrtl;

namespace {

StringRef edgeKindName(EdgeKind kind) {
  switch (kind) {
  case EdgeKind::Alias:
    return "Alias";
  case EdgeKind::Gate:
    return "Gate";
  case EdgeKind::InstanceIn:
    return "InstanceIn";
  case EdgeKind::InstanceOut:
    return "InstanceOut";
  }
  return "?";
}

/// The FModuleOp `value` lives in.
FModuleOp getParentModule(Value value) {
  if (isa<BlockArgument>(value))
    return cast<FModuleOp>(value.getParentBlock()->getParentOp());
  return value.getDefiningOp()->getParentOfType<FModuleOp>();
}

/// The clock operand of a supported root op, null otherwise. The `*_initial`
/// ref force/release variants have no clock, so they are not roots.
Value clockOperandOf(Operation *op) {
  if (auto fop = dyn_cast<RefForceOp>(op))
    return fop.getClock();
  if (auto rop = dyn_cast<RefReleaseOp>(op))
    return rop.getClock();
  if (auto reg = dyn_cast<RegOp>(op))
    return reg.getClockVal();
  if (auto regr = dyn_cast<RegResetOp>(op))
    return regr.getClockVal();
  if (auto gc = dyn_cast<ClockGateIntrinsicOp>(op))
    return gc.getInput();
  return Value();
}

} // namespace

//===----------------------------------------------------------------------===//
// GatedClockConversion: worklist analysis (no IR mutation)
//===----------------------------------------------------------------------===//

LogicalResult GatedClockConversion::addRoot(Operation *op) {
  Value clk = clockOperandOf(op);
  if (!clk)
    return op->emitError(
        "unsupported operation type for gated clock "
        "conversion; expected RefForceOp, RefReleaseOp, RegOp, "
        "RegResetOp or ClockGateIntrinsicOp");
  roots.emplace_back(op, clk);
  return success();
}

LogicalResult GatedClockConversion::analyzeFrom(ArrayRef<Value> seeds) {
  LLVM_DEBUG(llvm::dbgs() << "[analyzeFrom] " << seeds.size() << " seeds\n");
  SmallVector<Value> worklist(seeds.begin(), seeds.end());
  LogicalResult result = success();

  // Record the edge `srcClk` -> `dstClk` through `op` and enqueue the driver of
  // `srcClk`, looking through wire/node/cast aliases.
  auto pushIfFresh = [&](Value dstClk, Value srcClk, Operation *op,
                         EdgeKind kind) {
    if (!dstClk || !srcClk)
      return;
    LLVM_DEBUG(llvm::dbgs()
               << "  [pushIfFresh] edge kind=" << edgeKindName(kind) << "\n");
    Value baseClkDriver =
        getModuleScopedDriver(srcClk, /*lookThroughWires=*/true,
                              /*lookThroughNodes=*/true,
                              /*lookThroughCasts=*/true);
    // An undriven clock net has no source to thread a (base, enable) pair from,
    // and skipping it would break the invariant that every caller drives a
    // planned input pair. Report it while the IR is still untouched.
    if (!baseClkDriver) {
      mlir::emitError(srcClk.getLoc())
          << "gated clock conversion: this clock is not driven; run this "
             "utility after firrtl-expand-whens and firrtl-check-init";
      result = failure();
      return;
    }
    // `srcToDstClocks` is replayed forwards, from base clocks to users.
    if (kind != EdgeKind::Alias)
      srcToDstClocks[srcClk].push_back({dstClk, op, kind});
    if (baseClkDriver != srcClk)
      // Drives `srcClk` through wires/nodes/casts, so no op is needed.
      srcToDstClocks[baseClkDriver].push_back(
          {srcClk, nullptr, EdgeKind::Alias});
    if (!visited.insert(baseClkDriver).second)
      return;
    worklist.push_back(baseClkDriver);
  };

  // Backward DFS from leaf clock values to the base clock that drives them.
  while (!worklist.empty()) {
    Value clk = worklist.pop_back_val();
    // Case 1: clk is an input-port BlockArg (fan out to every caller).
    if (auto blockArg = dyn_cast<BlockArgument>(clk)) {
      auto mod = dyn_cast<FModuleOp>(blockArg.getOwner()->getParentOp());
      assert(mod &&
             mod.getPortDirection(blockArg.getArgNumber()) == Direction::In &&
             "expected input port of an FModuleOp");
      unsigned portIdx = blockArg.getArgNumber();
      auto *node = ig.lookup(mod);
      // Top-level module: this is the base clock, nothing else to traverse.
      if (node->uses().empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  top-level port, base clock\n");
        baseClks.push_back(clk);
        continue;
      }
      for (auto *use : node->uses()) {
        if (auto callerInst = dyn_cast<InstanceOp>(*use->getInstance()))
          pushIfFresh(clk, callerInst.getResult(portIdx), callerInst,
                      EdgeKind::InstanceIn);
        else
          use->getInstance()->emitError("can only handle InstanceOp");
      }
      continue;
    }
    auto *defOp = clk.getDefiningOp();

    // Case 2: clk is the result of a clock gate.
    if (auto gate = dyn_cast<ClockGateIntrinsicOp>(defOp)) {
      pushIfFresh(clk, gate.getInput(), gate, EdgeKind::Gate);
      continue;
    }

    // Case 3: clk is an instance result (descend into the referenced module).
    if (auto inst = dyn_cast<InstanceOp>(defOp)) {
      auto refMod = inst.getReferencedModule(ig);
      auto childMod = dyn_cast_or_null<FModuleOp>(refMod.getOperation());
      if (!childMod) {
        // External module: treat as base.
        LLVM_DEBUG(llvm::dbgs() << "  external module, base clock\n");
        baseClks.push_back(clk);
        continue;
      }
      unsigned portIdx = cast<OpResult>(clk).getResultNumber();
      pushIfFresh(clk, childMod.getBodyBlock()->getArgument(portIdx), inst,
                  EdgeKind::InstanceOut);
      continue;
    }
    if (isa<WireOp, NodeOp>(defOp)) {
      pushIfFresh(clk, clk, defOp, EdgeKind::Alias);
      continue;
    }

    // A clock mux is the post-ExpandWhens form of a multi-driver gated clock.
    // There is no single enable to sink through it, so say so rather than
    // silently leaving the gate in place. Remark only when a gate feeds the
    // mux, so that ordinary clock selection stays silent.
    if (auto mux = dyn_cast<MuxPrimOp>(defOp)) {
      Value inputs[] = {mux.getHigh(), mux.getLow()};
      if (llvm::any_of(inputs, [](Value v) {
            Value d = getModuleScopedDriver(v, /*lookThroughWires=*/true,
                                            /*lookThroughNodes=*/true,
                                            /*lookThroughCasts=*/true);
            return d && d.getDefiningOp<ClockGateIntrinsicOp>();
          }))
        mlir::emitRemark(mux.getLoc())
            << "gated clock conversion: clock selection is not supported; the "
               "clock gate feeding this mux was left in place";
    }

    // Any other op generating the clock is a base clock; stop tracing here.
    LLVM_DEBUG(llvm::dbgs() << "  base clock\n");
    baseClks.push_back(clk);
  }
  LLVM_DEBUG(llvm::dbgs() << "[analyzeFrom] " << baseClks.size()
                          << " base clocks\n");
  return result;
}

void GatedClockConversion::computeGatedClocks() {
  // Forward closure of every `Gate` edge over `srcToDstClocks`.
  SmallVector<Value> worklist;
  for (const auto &[src, edges] : srcToDstClocks)
    for (const auto &edge : edges)
      if (edge.kind == EdgeKind::Gate && gatedClocks.insert(edge.dst).second)
        worklist.push_back(edge.dst);

  // Iterating a DenseMap above is fine: the result is a set, not an order.
  while (!worklist.empty()) {
    Value clk = worklist.pop_back_val();
    auto it = srcToDstClocks.find(clk);
    if (it == srcToDstClocks.end())
      continue;
    for (const auto &edge : it->second)
      if (gatedClocks.insert(edge.dst).second)
        worklist.push_back(edge.dst);
  }
  LLVM_DEBUG(llvm::dbgs() << "[computeGatedClocks] " << gatedClocks.size()
                          << " gated clock values\n");
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: debug printing
//===----------------------------------------------------------------------===//

void GatedClockConversion::dump() const {
  llvm::dbgs() << "=== srcToDstClocks ===\n";
  for (const auto &[srcClk, dstList] : srcToDstClocks) {
    llvm::dbgs() << "Source clock: " << getParentModule(srcClk).getModuleName()
                 << "\n";
    srcClk.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    for (const auto &edge : dstList) {
      llvm::dbgs() << "  -> Destination clock: "
                   << getParentModule(edge.dst).getModuleName() << "\n";
      edge.dst.print(llvm::dbgs());
      llvm::dbgs() << " via op: ";
      if (edge.op)
        edge.op->print(llvm::dbgs());
      else
        llvm::dbgs() << "<alias>";
      llvm::dbgs() << " [" << edgeKindName(edge.kind) << "]\n";
    }
  }
  llvm::dbgs() << "=== Base clocks ===\n";
  for (const auto &baseClk : baseClks) {
    llvm::dbgs() << "  ";
    baseClk.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
  llvm::dbgs() << "======================\n";
}
