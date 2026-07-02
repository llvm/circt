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
#include "circt/Dialect/FIRRTL/FIRRTLEnums.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

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

/// Compute the gate's effective enable as `enable | test_enable` (or just
/// `enable`), materialising the OR after the gate op.
Value materializeGateEnable(ClockGateIntrinsicOp gate) {
  if (!gate.getTestEnable())
    return gate.getEnable();
  ImplicitLocOpBuilder b(gate.getLoc(), gate);
  return b.createOrFold<OrPrimOp>(gate.getEnable(), gate.getTestEnable());
}

/// Build the (baseClock, gateEnable) PortInfo pair for the given direction.
std::pair<PortInfo, PortInfo>
makeGatedClockPortInfos(MLIRContext *ctx, StringRef tag, Direction dir,
                        Location loc, Type clockType, Type u1Type) {
  return {PortInfo(StringAttr::get(ctx, ("_gatedClock_baseClock_" + tag).str()),
                   clockType, dir, /*symName=*/StringAttr(), loc),
          PortInfo(StringAttr::get(ctx, ("_gatedClock_enable_" + tag).str()),
                   u1Type, dir, /*symName=*/StringAttr(), loc)};
}

/// Return the parent FModuleOp of a given value: for a BlockArgument the module
/// owning its block, otherwise the module containing its defining op.
FModuleOp getParentModule(Value value) {
  if (isa<BlockArgument>(value))
    return cast<FModuleOp>(value.getParentBlock()->getParentOp());
  return value.getDefiningOp()->getParentOfType<FModuleOp>();
}

/// Return the clock operand of a registered root op, or null for ops that
/// have no clock (RefForceInitialOp / RefReleaseInitialOp).
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
  op->emitError("unsupported operation in gated clock conversion");
  return Value();
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

Value GatedClockConversion::getOrCreateConstU1One(FModuleOp mod) {
  auto it = constU1Cache.find(mod);
  if (it != constU1Cache.end())
    return it->second;

  // Create the constant at the beginning of the module body so it dominates
  // every possible use.
  ImplicitLocOpBuilder builder(mod.getLoc(), context);
  builder.setInsertionPointToStart(mod.getBodyBlock());
  Value constOne = builder.createOrFold<ConstantOp>(
      APSInt(APInt(1, 1, /*isSigned=*/false), /*isUnsigned=*/true));
  constU1Cache[mod] = constOne;
  return constOne;
}

void GatedClockConversion::connectMaterializedToInstancePorts(
    InstanceOp inst, unsigned clkPortIndex, unsigned enPortIndex,
    Value materializedClk, Value materializedEn, Location loc,
    bool checkUseEmpty) {
  ImplicitLocOpBuilder builder(loc, context);
  // Insert at the end of the instance's block so the materialized clock
  // dominates the connect.
  builder.setInsertionPointToEnd(inst->getBlock());

  auto clkPort = inst->getResult(clkPortIndex);
  if (!checkUseEmpty || clkPort.use_empty())
    MatchingConnectOp::create(builder, clkPort, materializedClk);

  auto enPort = inst->getResult(enPortIndex);
  if (!checkUseEmpty || enPort.use_empty()) {
    if (!materializedEn)
      materializedEn =
          getOrCreateConstU1One(inst->getParentOfType<FModuleOp>());
    MatchingConnectOp::create(builder, enPort, materializedEn);
  }
}

std::pair<unsigned, unsigned>
GatedClockConversion::insertPorts(FModuleOp mod, StringRef tag, Direction dir) {
  auto [baseInfo, enableInfo] = makeGatedClockPortInfos(
      mod.getContext(), tag, dir, mod.getLoc(), clockType, u1Type);

  // Append a fresh (base clock, enable) port pair at the end of the module
  // signature.
  unsigned baseIdx = mod.getNumPorts();
  unsigned enableIdx = baseIdx + 1;
  SmallVector<std::pair<unsigned, PortInfo>> newPorts = {{baseIdx, baseInfo},
                                                         {baseIdx, enableInfo}};
  mod.insertPorts(newPorts);

  auto *node = ig.lookup(mod);
  SmallVector<InstanceOp> oldInsts;
  for (auto *use : node->uses())
    if (auto i = dyn_cast<InstanceOp>(*use->getInstance()))
      oldInsts.push_back(i);

  for (auto oldInst : oldInsts) {
    auto cloneIface = oldInst.cloneWithInsertedPortsAndReplaceUses(newPorts);
    auto newInst = cast<InstanceOp>(cloneIface.getOperation());
    ig.replaceInstance(oldInst, newInst);
    opReplaceMap[oldInst] = newInst;
    // Record old-result → new-result value mappings before erasing `oldInst`,
    // so any of its result values cached in the analysis maps can be resolved
    // to the live instance. New result index = old index + number of inserted
    // ports at or before that index.
    for (unsigned i = 0, e = oldInst->getNumResults(); i < e; ++i)
      valueReplaceMap[oldInst->getResult(i)] = newInst->getResult(i);

    // Defer erasure until all transformations are complete.
    opsToErase.push_back(oldInst);
  }
  return {baseIdx, enableIdx};
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: worklist analysis (no IR mutation)
//===----------------------------------------------------------------------===//

LogicalResult GatedClockConversion::addRoot(Operation *op) {
  // Single-use: roots must not be added after the conversion has run. Create a
  // new instance for a fresh set of roots rather than reusing this one.
  assert(!hasRun && "GatedClockConversion::addRoot() called after run(); "
                    "create a new instance for a fresh set of roots");

  if (!isa<RefForceOp, RefReleaseOp, RefForceInitialOp, RefReleaseInitialOp,
           RegOp, RegResetOp>(op))
    return op->emitError("unsupported operation type for gated clock "
                         "conversion; expected RefForceOp, RefReleaseOp, "
                         "RefForceInitialOp, RefReleaseInitialOp, RegOp, or "
                         "RegResetOp");

  roots.push_back(op);
  return success();
}

void GatedClockConversion::analyzeFrom(ArrayRef<Value> seeds) {
  LLVM_DEBUG(llvm::dbgs() << "[analyzeFrom] " << seeds.size() << " seeds\n");
  SmallVector<Value> worklist(seeds.begin(), seeds.end());

  // Record the clock-flow relationship and push to the worklist. Looks through
  // wire/node/cast aliases to find the actual driver, recording both the direct
  // and transitive relationships in srcToDstClocks. `srcClk` is an operand of
  // `op` and `dstClk` is its result.
  auto pushIfFresh = [&](Value dstClk, Value srcClk, Operation *op,
                         EdgeKind kind) {
    if (!dstClk || !srcClk)
      return;
    LLVM_DEBUG(llvm::dbgs()
               << "  [pushIfFresh] edge kind=" << edgeKindName(kind) << "\n");
    // Record the `op` through which srcClk drives `dstClk`. This map is used to
    // backtrack the traversal from base clock to users.
    if (kind != EdgeKind::Alias)
      srcToDstClocks[srcClk].push_back({dstClk, op, kind});
    Value baseClkDriver =
        getModuleScopedDriver(srcClk, /*lookThroughWires=*/true,
                              /*lookThroughNodes=*/true,
                              /*lookThroughCasts=*/true);
    if (baseClkDriver != srcClk)
      // `baseClkDriver` drives `srcClk` through wires/nodes/casts. No op
      // needed.
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
          // `getModuleScopedDriver` will be used to follow the connect chain to
          // find the driving value for the input port, in the caller's module.
          pushIfFresh(clk, callerInst.getResult(portIdx), callerInst,
                      EdgeKind::InstanceIn);
        else
          use->getInstance()->emitError("can only handle InstanceOp");
      }
      continue;
    }
    auto *defOp = clk.getDefiningOp();

    // Case 2: clk is the result of a clock gate.
    if (isa<ClockGateIntrinsicOp>(defOp)) {
      pushIfFresh(clk, clockOperandOf(defOp), defOp, EdgeKind::Gate);
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

    // Any other op generating the clock is a base clock; stop tracing here.
    LLVM_DEBUG(llvm::dbgs() << "  base clock\n");
    baseClks.push_back(clk);
  }
  LLVM_DEBUG(llvm::dbgs() << "[analyzeFrom] " << baseClks.size()
                          << " base clocks\n");
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: rewriteRoot, run
//===----------------------------------------------------------------------===//

LogicalResult GatedClockConversion::rewriteRoot(Operation *op, Value base,
                                                Value enable) {
  if (!enable)
    return success();

  // RefForce/RefRelease: rebind the clock to the ungated base and fold the
  // enable into the predicate.
  if (auto fop = dyn_cast<RefForceOp>(op)) {
    fop.getClockMutable().assign(base);
    ImplicitLocOpBuilder b(fop.getLoc(), fop);
    fop.getPredicateMutable().assign(
        b.createOrFold<AndPrimOp>(fop.getPredicate(), enable));
    return success();
  }
  if (auto rop = dyn_cast<RefReleaseOp>(op)) {
    rop.getClockMutable().assign(base);
    ImplicitLocOpBuilder b(rop.getLoc(), rop);
    rop.getPredicateMutable().assign(
        b.createOrFold<AndPrimOp>(rop.getPredicate(), enable));
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
    return op->emitError("unsupported for gated clock conversion");

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
  if (!materializedEn) {
    clockEnablePairs[dstClk] = {materializedClk, {}};
    return;
  }
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
  if (!isa<BlockArgument>(materializedEn))
    builder.setInsertionPointAfterValue(materializedEn);
  MatchingConnectOp::create(builder, enWire, materializedEn);
  clockEnablePairs[dstClk] = {clockWire, enWire};
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
  clockEnablePairs[dstClk] = {materializedClk, gateEn};
}

std::pair<unsigned, unsigned> GatedClockConversion::findOrInsertGatedPorts(
    InstanceOp &inst, FModuleOp childMod, unsigned gatedClkIndex, Direction dir,
    Value materializedClk, Value materializedEn) {
  // Cache hit: ports already inserted for this (module, port) pair.
  auto modIt = modArgToMaterialized.find({childMod, gatedClkIndex});
  if (modIt != modArgToMaterialized.end())
    return {modIt->second.first, modIt->second.second};

  unsigned baseClkIndex = 0, enableIndex = 0;
  StringRef portName = childMod.getPortName(gatedClkIndex);

  if (dir == Direction::Out) {
    auto [cIdx, eIdx] = insertPorts(childMod, portName, Direction::Out);
    baseClkIndex = cIdx;
    enableIndex = eIdx;
    inst = liveOp(inst);
    auto newClkOut = childMod.getBodyBlock()->getArgument(baseClkIndex);
    auto enableOut = childMod.getBodyBlock()->getArgument(enableIndex);
    ImplicitLocOpBuilder builder(childMod.getLoc(), childMod);
    builder.setInsertionPointToEnd(childMod.getBodyBlock());
    MatchingConnectOp::create(builder, newClkOut, materializedClk);

    assert(materializedEn &&
           "unless this is a gated clock, no need to add output enable port");
    MatchingConnectOp::create(builder, enableOut, materializedEn);
  } else {
    auto [cIdx, eIdx] = insertPorts(childMod, portName, Direction::In);
    baseClkIndex = cIdx;
    enableIndex = eIdx;
    inst = liveOp(inst);
    connectMaterializedToInstancePorts(inst, baseClkIndex, enableIndex,
                                       materializedClk, materializedEn,
                                       inst.getLoc());
  }

  modArgToMaterialized[{childMod, gatedClkIndex}] = {baseClkIndex, enableIndex};
  return {baseClkIndex, enableIndex};
}

bool GatedClockConversion::materializeInstancePort(Direction dir,
                                                   InstanceOp inst,
                                                   Value dstClk, Value srcClk,
                                                   Value materializedClk,
                                                   Value materializedEn) {
  auto childMod =
      dyn_cast_or_null<FModuleOp>(inst.getReferencedModule(ig).getOperation());
  auto gatedClkIndex = cast<OpResult>(srcClk).getResultNumber();
  if (!materializedEn) {
    if (dir == Direction::Out) {
      clockEnablePairs[dstClk] = {dstClk, {}};
      return true;
    }
    // This instance has an ungated clock input, but a different instance of the
    // same module may have a gated clock. Iterate over all instances of
    // childMod to check whether any have gated inputs.
    auto *node = ig.lookup(childMod);
    bool allHaveNullEnable = true;

    for (auto *use : node->uses()) {
      if (auto instance = dyn_cast<InstanceOp>(*use->getInstance())) {
        auto instanceIn = instance.getResult(gatedClkIndex);
        auto it = clockEnablePairs.find(instanceIn);
        if (it == clockEnablePairs.end())
          // Not yet processed; wait for all instance inputs before deciding.
          return false;
        if (it->second.second) {
          allHaveNullEnable = false;
          break;
        }
      }
    }

    // All instances processed with null enable: ungated for all uses.
    if (allHaveNullEnable) {
      clockEnablePairs[dstClk] = {dstClk, {}};
      return true;
    }

    // Some instance has a gated clock and others are ungated. Add the enable
    // port to the ungated ones too, but connect it to a constant 1.
    materializedEn = getOrCreateConstU1One(inst->getParentOfType<FModuleOp>());
  }
  auto [baseClkIndex, enableIndex] = findOrInsertGatedPorts(
      inst, childMod, gatedClkIndex, dir, materializedClk, materializedEn);

  // An output port is exposed on the instance result side; an input port on the
  // child module's block-argument side. The materialized pair is recorded the
  // same way either way.
  Value baseVal, enVal;
  if (dir == Direction::Out) {
    baseVal = inst.getResult(baseClkIndex);
    enVal = inst.getResult(enableIndex);
  } else {
    baseVal = childMod.getBodyBlock()->getArgument(baseClkIndex);
    enVal = childMod.getBodyBlock()->getArgument(enableIndex);
  }
  assert(getParentModule(dstClk) == getParentModule(baseVal) &&
         "parent modules must match");
  clockEnablePairs[dstClk] = {baseVal, enVal};
  return true;
}

void GatedClockConversion::reinitMultiplyInstantiatedInput(
    Value liveSrc, Value materializedClk, Value materializedEn) {
  // liveSrc is a result of the live caller instance; drive the already-inserted
  // gated ports for the second (or later) caller of this module.
  auto inst = liveSrc.getDefiningOp<InstanceOp>();
  assert(inst);
  auto childMod =
      dyn_cast_or_null<FModuleOp>(inst.getReferencedModule(ig).getOperation());
  auto gatedClkIndex = cast<OpResult>(liveSrc).getResultNumber();
  auto modIt = modArgToMaterialized.find({childMod, gatedClkIndex});
  // No mapping means this is an output port. Output ports are materialized when
  // processing InstanceOut edges, not InstanceIn, so skip the reinit here and
  // let the normal InstanceOut path handle it.
  if (modIt == modArgToMaterialized.end()) {
    LLVM_DEBUG(llvm::dbgs() << "  no mapping for index " << gatedClkIndex
                            << ", skipping reinit (handled by InstanceOut)\n");
    return;
  }
  auto [newClkIndex, newEnIndex] = modIt->second;
  connectMaterializedToInstancePorts(inst, newClkIndex, newEnIndex,
                                     materializedClk, materializedEn,
                                     liveSrc.getLoc(), /*checkUseEmpty=*/true);
}

Value GatedClockConversion::findClockFlowCycle() const {
  // Pure DFS over the clock-flow graph with the classic three-color marking.
  // A Gray node reached again is an ancestor on the current path → back-edge →
  // feedback loop. The materialization worklist propagates forward from
  // `baseClks`, so we seed the search there; nodes unreachable from any base
  // clock are never materialized and cannot stall the forward pass.
  enum Color { White, Gray, Black };
  DenseMap<Value, Color> color;
  // Explicit stack of (node, next-edge-index) to avoid deep recursion.
  SmallVector<std::pair<Value, unsigned>> stack;

  for (auto base : baseClks) {
    if (color.lookup(base) != White)
      continue;
    color[base] = Gray;
    stack.push_back({base, 0});
    while (!stack.empty()) {
      auto &[node, idx] = stack.back();
      auto edgesIt = srcToDstClocks.find(node);
      if (edgesIt == srcToDstClocks.end() || idx >= edgesIt->second.size()) {
        color[node] = Black;
        stack.pop_back();
        continue;
      }
      Value dst = edgesIt->second[idx++].dst;
      switch (color.lookup(dst)) {
      case Gray:
        // Back-edge: `dst` is still on the active DFS path.
        return liveValue(dst);
      case White:
        color[dst] = Gray;
        stack.push_back({dst, 0});
        break;
      case Black:
        break; // already fully explored
      }
    }
  }
  return Value();
}

SmallVector<Value, 2> GatedClockConversion::processEdge(const ClockEdge &edge,
                                                        Value srcClk,
                                                        FModuleOp srcMod,
                                                        Value materializedClk,
                                                        Value materializedEn) {
  LLVM_DEBUG(llvm::dbgs() << "  edge kind=" << edgeKindName(edge.kind) << "\n");
  Value liveSrc = liveValue(srcClk);

  if (clockEnablePairs.count(edge.dst)) {
    // dst already done — for InstanceIn this means a multiply-instantiated
    // module whose ports were inserted by an earlier caller; re-init them.
    if (edge.kind == EdgeKind::InstanceIn)
      reinitMultiplyInstantiatedInput(liveSrc, materializedClk, materializedEn);
    return {};
  }

  // The destination is now reachable; visit it next.
  SmallVector<Value, 2> toEnqueue = {edge.dst};
  switch (edge.kind) {
  case EdgeKind::Alias:
    materializeAlias(edge.dst, srcMod, materializedClk, materializedEn);
    break;
  case EdgeKind::Gate:
    materializeGate(edge.gate(), edge.dst, materializedClk, materializedEn);
    break;
  case EdgeKind::InstanceIn:
    if (!materializeInstancePort(Direction::In, liveOp(edge.instance()),
                                 liveValue(edge.dst), liveSrc, materializedClk,
                                 materializedEn))
      // Multiply-instantiated module: wait for all instance inputs to be
      // materialized, then re-queue. If all instances have an ungated clock the
      // enable port is skipped; otherwise the ungated clocks get a const-1
      // enable port.
      toEnqueue.push_back(srcClk);
    break;
  case EdgeKind::InstanceOut:
    materializeInstancePort(Direction::Out, liveOp(edge.instance()), edge.dst,
                            liveValue(edge.dst), materializedClk,
                            materializedEn);
    break;
  }
  return toEnqueue;
}

void GatedClockConversion::materialize() {
  LLVM_DEBUG(llvm::dbgs() << "[materialize] " << baseClks.size()
                          << " base clocks\n");
  // Forward pass: propagate (base, enable) pairs from base clocks through the
  // clock flow graph built by analyzeFrom(), inserting ports as needed.
  //
  // The graph is acyclic here: run() calls findClockFlowCycle() before
  // materialize() and bails on any feedback loop. So every item that is
  // re-queued while blocked on an unmaterialized upstream is guaranteed to
  // unblock eventually.
  std::deque<Value> worklist(baseClks.begin(), baseClks.end());
  for (auto base : baseClks)
    clockEnablePairs[base] = {base, {}};

  // This is a BFS traversal, that tries to materialize all the ancestors before
  // moving on to the child nodes. For example, for a multiply instantiated
  // module, all its instances must be materialized, before the algorithm can
  // move into the module, which is required to infer if any of the instances of
  // a module have a gated clock input port.

  while (!worklist.empty()) {
    auto srcClk = worklist.front();
    worklist.pop_front();
    // Resolve to the live SSA value — a prior port insertion may have replaced
    // the defining instance.
    FModuleOp srcMod = getParentModule(liveValue(srcClk));

    // Check the base-clock, enable pair that drives `srcClk`. If its not yet
    // materialized, re-queue it.
    auto it = clockEnablePairs.find(srcClk);
    if (it == clockEnablePairs.end()) {
      worklist.push_back(srcClk);
      continue;
    }
    auto materializedClk = liveValue(it->second.first);
    auto materializedEn = liveValue(it->second.second);

    for (auto &edge : srcToDstClocks[srcClk])
      for (Value next :
           processEdge(edge, srcClk, srcMod, materializedClk, materializedEn))
        worklist.push_back(next);
  }
  LLVM_DEBUG(llvm::dbgs() << "[materialize] complete\n");
}

std::pair<SmallVector<Value>, SmallVector<std::pair<Operation *, Value>>>
GatedClockConversion::collectSeeds() {
  // Take ownership of the pending roots — `collectSeeds()` consumes them so the
  // caller's `addRoot` queue is empty afterwards (a subsequent visit can
  // collect a fresh batch without reprocessing — and without dereferencing ops
  // that may have been erased between visits).
  SmallVector<Operation *> pending;
  pending.swap(roots);

  // Collect each root's clock operand as a seed. Remember the
  // (op, original-clock) pairing: `materialized` is keyed by the original
  // analysis-era clock value, but the op's live clock operand may be remapped
  // to a replacement instance result before the rewrite runs.
  SmallVector<Value> seeds;
  SmallVector<std::pair<Operation *, Value>> rootClocks;
  seeds.reserve(pending.size());
  for (Operation *op : pending) {
    if (Value clk = clockOperandOf(op)) {
      seeds.push_back(clk);
      rootClocks.push_back({op, clk});
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[collectSeeds] " << seeds.size() << " seeds\n");
  return {seeds, rootClocks};
}

void GatedClockConversion::eliminateTemporaryWires() {
  // Eliminate temporary wires created during materialization by forwarding
  // their values directly. For each wire, expect exactly one connect writing to
  // it plus reads from known ops; if the write dominates every read, bypass the
  // wire by replacing its uses with the write source and erasing the wire.
  DenseMap<FModuleOp, mlir::DominanceInfo> dominanceInfo;
  for (auto wire : wireOps) {
    auto wireData = wire.getData();
    FModuleOp mod = wire->getParentOfType<FModuleOp>();
    if (!dominanceInfo.count(mod))
      dominanceInfo.try_emplace(mod, mod);
    auto &modDomInfo = dominanceInfo.find(mod)->second;

    FConnectLike writeConnect = {}; // Connect writing to the wire.
    bool cannotRemove = false;
    SmallVector<Operation *> wireReaders;

    for (auto *user : wireData.getUsers()) {
      if (auto connect = dyn_cast<MatchingConnectOp>(user)) {
        if (connect.getDest() == wireData) {
          // A second write means we can't safely forward; bail out.
          if (writeConnect) {
            cannotRemove = true;
            break;
          }
          writeConnect = connect;
          continue;
        }
      } else if (!isa<RegOp, RegResetOp, RefForceOp, RefReleaseOp, MuxPrimOp>(
                     user)) {
        // Unhandled user; can't optimize.
        cannotRemove = true;
        break;
      }
      wireReaders.push_back(user);
    }
    if (cannotRemove || !writeConnect)
      continue;

    // Bypass the wire if the write dominates every read.
    Value writeSource = writeConnect.getSrc();
    if (llvm::all_of(wireReaders, [&](Operation *user) {
          return modDomInfo.dominates(writeConnect, user);
        })) {
      wireData.replaceAllUsesWith(writeSource);
      // The read connect is implicitly removed by replaceAllUsesWith updating
      // its source operand.
      writeConnect.erase();
      wire.erase();
    }
  }
}

LogicalResult GatedClockConversion::run() {
  LLVM_DEBUG(llvm::dbgs() << "===== GatedClockConversion::run() =====\n");
  // Single-use: mark this instance as run so a later addRoot() is rejected.
  hasRun = true;

  auto [seeds, rootClocks] = collectSeeds();

  if (seeds.empty())
    return success();
  context = seeds[0].getContext();
  clockType = ClockType::get(context);
  u1Type = UIntType::get(context, 1);

  // Phase 1: pure analysis.
  LLVM_DEBUG(llvm::dbgs() << "--- Phase 1: Analysis ---\n");
  analyzeFrom(seeds);
  LLVM_DEBUG(dump());

  // Cyclic clock flow cannot be materialized. Detect it on the pure analysis
  // graph *before* any IR mutation, so that on a cycle we leave the input
  // untouched and complete successfully (a no-op conversion + warning) rather
  // than failing the pass with a partially-mutated module.
  if (Value cyc = findClockFlowCycle()) {
    mlir::emitWarning(cyc.getLoc())
        << "cyclic clock dependency: skipping gated-clock conversion "
           "(clock flow contains a feedback loop); module left unchanged";
    return success();
  }

  // Phase 2: per-root materialisation + IR rewrite. The clock-flow graph is
  // acyclic (checked above), so materialization always succeeds.
  LLVM_DEBUG(llvm::dbgs() << "--- Phase 2: Materialization ---\n");
  materialize();

  LLVM_DEBUG(llvm::dbgs() << "--- Phase 2b: Root Rewriting ---\n");
  for (auto &[op, clk] : rootClocks) {
    assert(clockEnablePairs.find(clk) != clockEnablePairs.end());
    // Resolve to live values: the materialized (base, enable) pair may have
    // been instance results that were since replaced by port insertion.
    auto materializedClk = liveValue(clockEnablePairs[clk].first);
    auto materializedEn = liveValue(clockEnablePairs[clk].second);
    if (failed(rewriteRoot(op, materializedClk, materializedEn)))
      return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "--- Phase 2c: Wire Elimination ---\n");
  eliminateTemporaryWires();

  // Phase 3: erase all replaced instances now that transformations are
  // complete.
  LLVM_DEBUG(llvm::dbgs() << "--- Phase 3: Cleanup (" << opsToErase.size()
                          << " ops) ---\n");
  for (auto oldInst : opsToErase)
    oldInst.erase();
  opsToErase.clear();

  LLVM_DEBUG(llvm::dbgs() << "===== run() complete =====\n");
  return success();
}

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
