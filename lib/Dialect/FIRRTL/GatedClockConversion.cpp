//===- GatedClockConversion.cpp - Gated clock conversion -----------------===//
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
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

#define DEBUG_TYPE "firrtl-gated-clock-conversion"

using namespace circt;
using namespace firrtl;

namespace {

/// Internal enum to specify which ports to insert in insertOrReusePort.
enum class PortInsertMode {
  Both,      ///< Insert both base clock and enable ports
  ClockOnly, ///< Insert only the base clock port
  EnableOnly ///< Insert only the enable port
};

/// Compute the gate's effective enable as `enable | test_enable` (or just
/// `enable`), materialising the OR after the gate op.
static Value materializeGateEnable(ClockGateIntrinsicOp gate) {
  if (!gate.getTestEnable())
    return gate.getEnable();
  ImplicitLocOpBuilder b(gate.getLoc(), gate);
  b.setInsertionPointAfter(gate);
  return b.createOrFold<OrPrimOp>(gate.getEnable(), gate.getTestEnable());
}

/// Build the (baseClock, gateEnable) PortInfo pair for the given direction.
static std::pair<PortInfo, PortInfo>
makeGatedClockPortInfos(MLIRContext *ctx, StringRef tag, Direction dir,
                        Location loc, Type clockType, Type u1Type) {
  return {PortInfo(StringAttr::get(ctx, ("_gatedClock_baseClock_" + tag).str()),
                   clockType, dir, /*symName=*/StringAttr(), loc),
          PortInfo(StringAttr::get(ctx, ("_gatedClock_enable_" + tag).str()),
                   u1Type, dir, /*symName=*/StringAttr(), loc)};
}

/// Materialize a 1-bit constant 1 at `b`'s insertion point.
static Value constU1One(ImplicitLocOpBuilder &b) {
  return b.createOrFold<ConstantOp>(
      APSInt(APInt(1, 1, /*isSigned=*/false), /*isUnsigned=*/true));
}

/// Return the clock operand of a registered root op, or null for ops that
/// have no clock (RefForceInitialOp / RefReleaseInitialOp).
static Value clockOperandOf(Operation *op) {
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

/// Result of scanning existing ports for a (clock, enable) pair match.
struct ExistingPortMatch {
  bool foundClk = false, foundEn = false;
  unsigned clkIndex = 0, enIndex = 0;
};

/// Check if existing ports already match the materialized clock or enable
/// values. Used for both output port reuse (checking module port arguments) and
/// input port reuse (checking instance port results).
static ExistingPortMatch checkExistingPorts(
    unsigned numPorts, llvm::function_ref<Value(unsigned)> getPortValue,
    Value materializedClk, Value materializedEn, Type clockType, Type u1Type) {
  ExistingPortMatch result;
  for (unsigned i = 0; i < numPorts; ++i) {
    auto portValue = getPortValue(i);
    if (!portValue)
      continue;
    if (portValue.getType() != clockType && portValue.getType() != u1Type)
      continue;
    if (auto driver = getDriverFromConnect(portValue)) {
      if (materializedEn && driver == materializedEn) {
        result.foundEn = true;
        result.enIndex = i;
      }
      if (driver == materializedClk) {
        result.foundClk = true;
        result.clkIndex = i;
      }
    }
    if (result.foundClk && (!materializedEn || result.foundEn))
      break;
  }
  return result;
}

} // namespace

//===----------------------------------------------------------------------===//
// GatedClockConversion: gate-enable cache + lazy port-pair insertion
//===----------------------------------------------------------------------===//

Value GatedClockConversion::gateEnableOf(ClockGateIntrinsicOp gate) {
  auto it = gateEnableCache.find(gate);
  if (it != gateEnableCache.end())
    return it->second;
  Value v = materializeGateEnable(gate);
  gateEnableCache[gate] = v;
  return v;
}

void GatedClockConversion::connectMaterializedToInstancePorts(
    InstanceOp inst, unsigned clkPortIndex, unsigned enPortIndex,
    Value materializedClk, Value materializedEn, Location loc,
    bool checkUseEmpty) {
  ImplicitLocOpBuilder builder(loc, context);
  // To ensure the materializedClk dominates the connect, create it at the end
  // of the block containing instance.
  builder.setInsertionPointToEnd(inst->getBlock());

  auto clkPort = inst->getResult(clkPortIndex);
  if (!checkUseEmpty || clkPort.use_empty())
    MatchingConnectOp::create(builder, clkPort, materializedClk);

  if (materializedEn) {
    auto enPort = inst->getResult(enPortIndex);
    if (!checkUseEmpty || enPort.use_empty())
      MatchingConnectOp::create(builder, enPort, materializedEn);
  }
}

GatedClockConversion::PortPair GatedClockConversion::insertOrReusePort(
    FModuleOp mod, StringRef tag, Direction dir,
    std::optional<unsigned> existingClockIdx,
    std::optional<unsigned> existingEnableIdx) {
  unsigned baseIdx;
  unsigned enableIdx;

  // Infer the mode based on which optional indices are provided
  PortInsertMode mode;
  if (existingClockIdx.has_value() && existingEnableIdx.has_value()) {
    // Both indices provided: reuse both, no insertion needed
    baseIdx = *existingClockIdx;
    enableIdx = *existingEnableIdx;
    return {baseIdx, enableIdx};
  }
  if (existingClockIdx.has_value()) {
    // Only clock index provided: reuse clock, insert enable only
    mode = PortInsertMode::EnableOnly;
    baseIdx = *existingClockIdx;
  } else if (existingEnableIdx.has_value()) {
    // Only enable index provided: reuse enable, insert clock only
    mode = PortInsertMode::ClockOnly;
    enableIdx = *existingEnableIdx;
  } else {
    // Neither provided: insert both
    mode = PortInsertMode::Both;
  }

  auto [baseInfo, enableInfo] = makeGatedClockPortInfos(
      mod.getContext(), tag, dir, mod.getLoc(), clockType, u1Type);

  SmallVector<std::pair<unsigned, PortInfo>> newPorts;
  switch (mode) {
  case PortInsertMode::EnableOnly:
    // Only insert the enable port
    enableIdx = mod.getNumPorts();
    mod.insertPorts({{enableIdx, enableInfo}});
    newPorts.push_back({enableIdx, enableInfo});
    break;
  case PortInsertMode::ClockOnly:
    // Only insert the base clock port
    baseIdx = mod.getNumPorts();
    mod.insertPorts({{baseIdx, baseInfo}});
    newPorts.push_back({baseIdx, baseInfo});
    break;
  case PortInsertMode::Both:
    // Insert both base and enable ports
    baseIdx = mod.getNumPorts();
    enableIdx = baseIdx + 1;
    mod.insertPorts({{baseIdx, baseInfo}, {baseIdx, enableInfo}});
    newPorts.push_back({baseIdx, baseInfo});
    newPorts.push_back({enableIdx, enableInfo});
    break;
  }

  auto *node = ig.lookup(mod);
  SmallVector<InstanceOp> oldInsts;
  for (auto *use : node->uses())
    if (auto i = dyn_cast<InstanceOp>(*use->getInstance()))
      oldInsts.push_back(i);

  for (auto oldInst : oldInsts) {
    auto cloneIface = oldInst.cloneWithInsertedPortsAndReplaceUses(newPorts);
    auto newInst = cast<InstanceOp>(cloneIface.getOperation());
    ig.replaceInstance(oldInst, newInst);
    instReplaceMap[oldInst] = newInst;
    opReplaceMap[oldInst.getOperation()] = newInst.getOperation();
    // Record old-result → new-result value mappings before erasing `oldInst`,
    // so any of its result values cached in the analysis maps can be resolved
    // to the live instance.  New result index = old index + number of inserted
    // ports at or before that index (mirrors
    // replaceUsesRespectingInsertedPorts).
    for (unsigned i = 0, e = oldInst->getNumResults(); i < e; ++i)
      valueReplaceMap[oldInst->getResult(i)] = newInst->getResult(i);

    // Defer erasure until all transformations are complete
    opsToErase.push_back(oldInst);
  }
  return {baseIdx, enableIdx};
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: worklist analysis (no IR mutation)
//===----------------------------------------------------------------------===//

void GatedClockConversion::clearAnalysis() {
  visited.clear();
  srcToDstClocks.clear();
  baseClks.clear();
  modArgToMaterialized.clear();
  wireOps.clear();
}

void GatedClockConversion::analyzeFrom(ArrayRef<Value> seeds) {
  clearAnalysis();
  SmallVector<Value> worklist(seeds.begin(), seeds.end());

  // Helper to record the clock flow relationship and push to worklist.
  // Looks through wire/node/cast aliases to find the actual driver, then
  // records both the direct and transitive relationships in srcToDstClocks.
  // `srcClk` is an operand of `op` and `dstClk` is the result of it.
  auto pushIfFresh = [&](Value dstClk, Value srcClk, Operation *op,
                         EdgeKind kind) {
    if (!dstClk || !srcClk)
      return;
    // Record the `op` through which srcClk drives `dstClk`. This map will be
    // used to backtrack the traversal from base clock to users.
    srcToDstClocks[srcClk].push_back({dstClk, op, kind});
    Value baseClkDriver =
        getModuleScopedDriver(srcClk, /*lookThroughWires=*/true,
                              /*lookThroughNodes=*/true,
                              /*lookThroughCasts=*/true);
    if (baseClkDriver) {
      // `baseClkDriver` drives `srcClk`, through wires/nodes/casts. No op
      // needed.
      srcToDstClocks[baseClkDriver].push_back(
          {srcClk, nullptr, EdgeKind::Alias});
    } else
      baseClkDriver = srcClk;
    // Only add to worklist if not already visited.
    if (!visited.insert(baseClkDriver).second)
      return;
    worklist.push_back(baseClkDriver);
  };

  // This is a (backward) DFS traversal from leaf clock values to the base clock
  // that drives them.
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
      for (auto *use : node->uses()) {
        // Can only handle InstanceOp.
        if (auto callerInst = dyn_cast<InstanceOp>(*use->getInstance())) {
          // The input-port "operand" at a caller InstanceOp is exposed as
          // the corresponding SSA result (drivable via firrtl.connect).
          // `getModuleScopedDriver` follows the connect chain to find the
          // driving value in the caller's module.
          pushIfFresh(clk, callerInst.getResult(portIdx), callerInst,
                      EdgeKind::InstanceIn);
        }
      }
      // If top level module, then we have reached the base clock, nothing else
      // to traverse from here.
      if (node->uses().empty())
        baseClks.push_back(clk);
      continue;
    }
    auto *defOp = clk.getDefiningOp();

    // Case 2: clk is the result of a clock gate.
    if (isa_and_nonnull<ClockGateIntrinsicOp>(defOp)) {
      pushIfFresh(clk, clockOperandOf(defOp), defOp, EdgeKind::Gate);
      continue;
    }

    // Case 3: clk is an instance result (descend into the referenced module).
    if (auto inst = dyn_cast_or_null<InstanceOp>(defOp)) {
      auto refMod = inst.getReferencedModule(ig);
      auto childMod = dyn_cast_or_null<FModuleOp>(refMod.getOperation());
      if (!childMod) {
        // external module: treat as base
        baseClks.push_back(clk);
        continue;
      }
      auto *childBody = childMod.getBodyBlock();
      unsigned portIdx = cast<OpResult>(clk).getResultNumber();
      pushIfFresh(clk, childBody->getArgument(portIdx), inst,
                  EdgeKind::InstanceOut);
      continue;
    }

    // Case 4: clk flows through a wire/node/cast alias (e.g. a register
    // clocked by a wire that is connected to a gated clock).  Look through the
    // alias to the real driver and record the alias relationship so the
    // forward pass materializes the (base, enable) pair onto `clk`.
    if (Value driver = getModuleScopedDriver(clk, /*lookThroughWires=*/true,
                                             /*lookThroughNodes=*/true,
                                             /*lookThroughCasts=*/true)) {
      if (driver != clk) {
        srcToDstClocks[driver].push_back({clk, nullptr, EdgeKind::Alias});
        if (visited.insert(driver).second)
          worklist.push_back(driver);
        continue;
      }
    }

    // Base case: clk is a base clock (no further source to trace).
    baseClks.push_back(clk);
  }
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: rewriteRoot, run
//===----------------------------------------------------------------------===//

LogicalResult GatedClockConversion::rewriteRoot(Operation *op, Value base,
                                                Value enable) {
  if (!enable)
    return success();
  if (auto fop = dyn_cast<RefForceOp>(op)) {
    fop.getClockMutable().assign(base);
    ImplicitLocOpBuilder b(fop.getLoc(), fop);
    Value newPred = b.createOrFold<AndPrimOp>(fop.getPredicate(), enable);
    fop.getPredicateMutable().assign(newPred);
    return success();
  }

  if (auto rop = dyn_cast<RefReleaseOp>(op)) {
    rop.getClockMutable().assign(base);
    ImplicitLocOpBuilder b(rop.getLoc(), rop);
    Value newPred = b.createOrFold<AndPrimOp>(rop.getPredicate(), enable);
    rop.getPredicateMutable().assign(newPred);
    return success();
  }

  if (isa<RefForceInitialOp, RefReleaseInitialOp>(op))
    return success();

  Value regData;
  if (auto reg = dyn_cast<RegOp>(op))
    regData = reg.getData();
  else if (auto regr = dyn_cast<RegResetOp>(op))
    regData = regr.getData();
  else
    return success();

  // Rebind the register clock to the ungated base.
  op->setOperand(0, base);

  // Wrap the unique connect driving regData with mux(enable, RHS, regData)
  // so the register holds when the clock gate is closed.
  for (auto &use : llvm::make_early_inc_range(regData.getUses())) {
    auto fconn = dyn_cast<FConnectLike>(use.getOwner());
    if (!fconn || fconn.getDest() != regData)
      continue;
    ImplicitLocOpBuilder b(fconn.getLoc(), fconn);
    Value newRhs = b.createOrFold<MuxPrimOp>(enable, fconn.getSrc(), regData);
    fconn->setOperand(1, newRhs);
    break;
  }
  return success();
}

void GatedClockConversion::materializeAlias(Value dstClk, FModuleOp srcMod,
                                            Value materializedClk,
                                            Value materializedEn) {
  auto liveDst = liveValue(dstClk);
  // Create temporary wires at the top of srcMod to carry (base, enable).
  // eliminateTemporaryWires() will forward them away when safe.
  auto builder = ImplicitLocOpBuilder::atBlockBegin(liveDst.getLoc(),
                                                    srcMod.getBodyBlock());
  auto createWire = [&](Type type) {
    auto w = WireOp::create(builder, type);
    wireOps.push_back(w);
    return w.getData();
  };
  auto clockWire = createWire(clockType);
  auto enWire = createWire(u1Type);
  if (!isa<BlockArgument>(materializedClk))
    builder.setInsertionPointAfterValue(materializedClk);
  MatchingConnectOp::create(builder, clockWire, materializedClk);
  if (materializedEn && !isa<BlockArgument>(materializedEn))
    builder.setInsertionPointAfterValue(materializedEn);
  MatchingConnectOp::create(
      builder, enWire, materializedEn ? materializedEn : constU1One(builder));
  materialized[dstClk] = {clockWire, enWire};
}

void GatedClockConversion::materializeGate(ClockGateIntrinsicOp gate,
                                           Value dstClk, Value materializedClk,
                                           Value materializedEn) {
  auto gateEn = gateEnableOf(gate);
  if (materializedEn) {
    // AND upstream enable with this gate's enable so the register holds
    // whenever any gate in the chain is closed.
    ImplicitLocOpBuilder builder(gate.getLoc(), gate);
    builder.setInsertionPointAfterValue(liveValue(dstClk));
    gateEn = builder.createOrFold<AndPrimOp>(materializedEn, gateEn);
  }
  // Propagate the true ungated base clock through cascaded gates.
  materialized[dstClk] = {materializedClk, gateEn};
}

GatedClockConversion::PortPair GatedClockConversion::findOrInsertGatedPorts(
    InstanceOp &inst, FModuleOp childMod, unsigned gatedClkIndex, Direction dir,
    Value materializedClk, Value materializedEn) {
  // Cache hit: ports already inserted for this (module, port) pair.
  auto modIt = modArgToMaterialized.find({childMod, gatedClkIndex});
  if (modIt != modArgToMaterialized.end())
    return {modIt->second.first, modIt->second.second};

  unsigned baseClkIndex = 0, enableIndex = 0;
  StringRef portName = childMod.getPortName(gatedClkIndex);

  if (dir == Direction::Out) {
    auto getPortValue = [&](unsigned i) -> Value {
      if (childMod.getPortDirection(i) != Direction::Out)
        return Value();
      return childMod.getBodyBlock()->getArgument(i);
    };
    auto match =
        checkExistingPorts(childMod.getNumPorts(), getPortValue,
                           materializedClk, materializedEn, clockType, u1Type);
    if (!match.foundClk || !match.foundEn) {
      auto [cIdx, eIdx] = insertOrReusePort(
          childMod, portName, Direction::Out,
          match.foundClk ? std::optional<unsigned>(match.clkIndex)
                         : std::nullopt,
          match.foundEn ? std::optional<unsigned>(match.enIndex)
                        : std::nullopt);
      baseClkIndex = cIdx;
      enableIndex = eIdx;
      inst = instReplaceMap[inst];
    } else {
      baseClkIndex = match.clkIndex;
      enableIndex = match.enIndex;
    }
    auto newClkOut = childMod.getBodyBlock()->getArgument(baseClkIndex);
    auto enableOut = childMod.getBodyBlock()->getArgument(enableIndex);
    ImplicitLocOpBuilder builder(childMod.getLoc(), childMod);
    builder.setInsertionPointToEnd(childMod.getBodyBlock());
    MatchingConnectOp::create(builder, newClkOut, materializedClk);
    if (materializedEn)
      MatchingConnectOp::create(builder, enableOut, materializedEn);
  } else {
    auto getPortValue = [&](unsigned i) -> Value { return inst->getResult(i); };
    auto match =
        checkExistingPorts(inst->getNumResults(), getPortValue, materializedClk,
                           materializedEn, clockType, u1Type);
    if (!match.foundClk || !match.foundEn) {
      auto [cIdx, eIdx] = insertOrReusePort(
          childMod, portName, Direction::In,
          match.foundClk ? std::optional<unsigned>(match.clkIndex)
                         : std::nullopt,
          match.foundEn ? std::optional<unsigned>(match.enIndex)
                        : std::nullopt);
      baseClkIndex = cIdx;
      enableIndex = eIdx;
      inst = instReplaceMap[inst];
    } else {
      baseClkIndex = match.clkIndex;
      enableIndex = match.enIndex;
    }
    connectMaterializedToInstancePorts(inst, baseClkIndex, enableIndex,
                                       materializedClk, materializedEn,
                                       inst.getLoc());
  }

  modArgToMaterialized[{childMod, gatedClkIndex}] = {baseClkIndex, enableIndex};
  return {baseClkIndex, enableIndex};
}

void GatedClockConversion::materializeInstancePort(
    Direction dir, InstanceOp inst, Value dstClk, Value liveAnchor,
    Value materializedClk, Value materializedEn) {
  auto childMod = dyn_cast_or_null<FModuleOp>(inst.getReferencedModule(ig));
  auto gatedClkIndex = cast<OpResult>(liveAnchor).getResultNumber();
  auto [baseClkIndex, enableIndex] = findOrInsertGatedPorts(
      inst, childMod, gatedClkIndex, dir, materializedClk, materializedEn);
  if (dir == Direction::Out)
    materialized[dstClk] = {inst.getResult(baseClkIndex),
                            inst.getResult(enableIndex)};
  else
    materialized[dstClk] = {childMod.getBodyBlock()->getArgument(baseClkIndex),
                            childMod.getBodyBlock()->getArgument(enableIndex)};
}

void GatedClockConversion::reinitMultiplyInstantiatedInput(
    Value liveSrc, Value materializedClk, Value materializedEn) {
  // liveSrc is a result of the live caller instance; drive the already-inserted
  // gated ports for the second (or later) caller of this module.
  auto inst = dyn_cast_or_null<InstanceOp>(liveSrc.getDefiningOp());
  if (!inst)
    return;
  auto childMod = dyn_cast_or_null<FModuleOp>(inst.getReferencedModule(ig));
  auto gatedClkIndex = cast<OpResult>(liveSrc).getResultNumber();
  auto modIt = modArgToMaterialized.find({childMod, gatedClkIndex});
  assert(modIt != modArgToMaterialized.end());
  auto [newClkIndex, newEnIndex] = modIt->second;
  connectMaterializedToInstancePorts(inst, newClkIndex, newEnIndex,
                                     materializedClk, materializedEn,
                                     liveSrc.getLoc(), /*checkUseEmpty=*/true);
}

void GatedClockConversion::materialize() {
  // Forward pass: propagate (base, enable) pairs from base clocks through the
  // clock flow graph built by analyzeFrom(), inserting ports as needed.
  std::deque<Value> worklist(baseClks.begin(), baseClks.end());
  for (auto base : baseClks)
    materialized[base] = {base, {}};

  while (!worklist.empty()) {
    auto srcClk = worklist.front();
    worklist.pop_front();
    // Resolve to the live SSA value — a prior port insertion may have replaced
    // the defining instance.
    auto liveSrc = liveValue(srcClk);
    FModuleOp srcMod;
    if (auto blockArg = dyn_cast<BlockArgument>(liveSrc))
      srcMod = cast<FModuleOp>(liveSrc.getParentBlock()->getParentOp());
    else if (auto *def = liveSrc.getDefiningOp())
      srcMod = def->getParentOfType<FModuleOp>();

    auto it = materialized.find(srcClk);
    if (it == materialized.end()) {
      // Not yet materialized; re-queue until its upstream is done.
      worklist.push_back(srcClk);
      continue;
    }
    auto materializedClk = liveValue(it->second.first);
    auto materializedEn = liveValue(it->second.second);

    for (auto &edge : srcToDstClocks[srcClk]) {
      liveSrc = liveValue(srcClk);
      if (materialized.count(edge.dst)) {
        // dst already done — for InstanceIn this means a multiply-instantiated
        // module whose ports were inserted by an earlier caller; re-init them.
        if (edge.kind == EdgeKind::InstanceIn)
          reinitMultiplyInstantiatedInput(liveSrc, materializedClk,
                                          materializedEn);
        continue;
      }
      worklist.push_back(edge.dst);
      switch (edge.kind) {
      case EdgeKind::Alias:
        materializeAlias(edge.dst, srcMod, materializedClk, materializedEn);
        break;
      case EdgeKind::Gate:
        materializeGate(cast<ClockGateIntrinsicOp>(edge.op), edge.dst,
                        materializedClk, materializedEn);
        break;
      case EdgeKind::InstanceIn: {
        auto inst = cast<InstanceOp>(liveOp(edge.op));
        materializeInstancePort(Direction::In, inst, edge.dst, liveSrc,
                                materializedClk, materializedEn);
        break;
      }
      case EdgeKind::InstanceOut: {
        auto liveDst = liveValue(edge.dst);
        auto inst = cast<InstanceOp>(liveOp(edge.op));
        materializeInstancePort(Direction::Out, inst, edge.dst, liveDst,
                                materializedClk, materializedEn);
        break;
      }
      }
    }
  }
}

std::pair<SmallVector<Value>, SmallVector<std::pair<Operation *, Value>>>
GatedClockConversion::collectSeeds() {
  // Take ownership of the pending roots — `collectSeeds()` consumes them so
  // the caller's `addRoot` queue is empty afterwards (a subsequent visit can
  // collect a fresh batch without reprocessing — and without dereferencing
  // ops that may have been erased between visits).
  SmallVector<Operation *> pending;
  pending.swap(roots);

  // Collect each root's clock operand as a seed.
  // Remember the (op, original-clock) pairing: `materialized` is keyed by the
  // original analysis-era clock value, but the op's live clock operand may be
  // remapped to a replacement instance result before the rewrite runs.
  SmallVector<Value> seeds;
  SmallVector<std::pair<Operation *, Value>> rootClocks;
  seeds.reserve(pending.size());
  for (Operation *op : pending)
    if (Value clk = clockOperandOf(op)) {
      seeds.push_back(clk);
      rootClocks.push_back({op, clk});
    }

  return {seeds, rootClocks};
}

void GatedClockConversion::eliminateTemporaryWires() {
  // Eliminate temporary wires created during materialization by forwarding
  // their values directly. For each wire, check if there are exactly two uses:
  // one connect writing to the wire and one connect reading from it. If the
  // write dominates the read, bypass the wire by replacing all uses of the wire
  // with the write source, then erase the wire and write connect.
  DenseMap<FModuleOp, mlir::DominanceInfo> dominanceInfo;
  for (auto wire : wireOps) {
    auto wireData = wire.getData();
    FModuleOp mod = wire->getParentOfType<FModuleOp>();
    auto domIter = dominanceInfo.find(mod);
    if (domIter == dominanceInfo.end()) {
      dominanceInfo[mod] = mlir::DominanceInfo(mod);
    }
    auto &modDomInfo = dominanceInfo[mod];

    // Find all connect operations using this wire.
    FConnectLike writeConnect = {}; // Connect writing to wire (wire is dest)
    FConnectLike readConnect = {};  // Connect reading from wire (wire is src)

    for (auto *user : wireData.getUsers()) {
      if (auto connect = dyn_cast<FConnectLike>(user)) {
        if (connect.getDest() == wireData) {
          // Found a write to the wire. If we already have one, bail out.
          if (writeConnect) {
            writeConnect = {};
            break;
          }
          writeConnect = connect;
        } else if (connect.getSrc() == wireData) {
          // Found a read from the wire. If we already have one, bail out.
          if (readConnect) {
            readConnect = {};
            break;
          }
          readConnect = connect;
        }
      } else
        // Wire has a non-connect user; can't optimize.
        break;
    }
    // Skip wires that don't have exactly one write and one read.
    if (!readConnect || !writeConnect)
      continue;

    // Optimize: bypass the wire if the write dominates the read.
    // Replace all uses of the wire with the write source.
    Value writeSource = writeConnect.getSrc();
    if (modDomInfo.dominates(writeConnect, readConnect)) {
      wireData.replaceAllUsesWith(writeSource);
      // Erase the write connect and the wire itself. The read connect is
      // implicitly removed by replaceAllUsesWith updating its source operand.
      writeConnect.erase();
      wire.erase();
    }
  }
}

LogicalResult GatedClockConversion::run() {
  // Collect clock operands from pending roots.
  auto [seeds, rootClocks] = collectSeeds();

  if (seeds.empty())
    return success();
  context = seeds[0].getContext();
  clockType = ClockType::get(context);
  u1Type = UIntType::get(context, 1);

  // Phase 1: pure analysis.
  analyzeFrom(seeds);
  LLVM_DEBUG(dump());
  // Phase 2: per-root materialisation + IR rewrite.
  materialize();
  for (auto &[op, clk] : rootClocks) {
    assert(materialized.find(clk) != materialized.end());
    // Resolve to live values: the materialized (base, enable) pair may have
    // been instance results that were since replaced by port insertion.
    auto materializedClk = liveValue(materialized[clk].first);
    auto materializedEn = liveValue(materialized[clk].second);
    if (failed(rewriteRoot(op, materializedClk, materializedEn)))
      return failure();
  }
  // Eliminate temporary wires created during materialization.
  eliminateTemporaryWires();

  // Phase 3: erase all replaced instances now that transformations are
  // complete.
  for (auto oldInst : opsToErase)
    oldInst.erase();
  opsToErase.clear();

  return success();
}

void GatedClockConversion::dump() const {
  llvm::dbgs() << "=== srcToDstClocks ===\n";
  for (const auto &[srcClk, dstList] : srcToDstClocks) {
    llvm::dbgs() << "Source clock: ";
    srcClk.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    for (const auto &edge : dstList) {
      llvm::dbgs() << "  -> Destination clock: ";
      edge.dst.print(llvm::dbgs());
      llvm::dbgs() << " via op: ";
      if (edge.op)
        edge.op->print(llvm::dbgs());
      else
        llvm::dbgs() << "<alias>";
      llvm::dbgs() << " [";
      switch (edge.kind) {
      case EdgeKind::Alias:
        llvm::dbgs() << "Alias";
        break;
      case EdgeKind::Gate:
        llvm::dbgs() << "Gate";
        break;
      case EdgeKind::InstanceIn:
        llvm::dbgs() << "InstanceIn";
        break;
      case EdgeKind::InstanceOut:
        llvm::dbgs() << "InstanceOut";
        break;
      }
      llvm::dbgs() << "]\n";
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
