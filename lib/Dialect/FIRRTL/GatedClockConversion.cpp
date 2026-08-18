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

/// The gate's effective enable, `enable | test_enable` or just `enable`.
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
// GatedClockConversion: the plan's value model
//===----------------------------------------------------------------------===//

void GatedClockConversion::MatRef::print(llvm::raw_ostream &os) const {
  switch (kind) {
  case Kind::None:
    os << "<none>";
    return;
  case Kind::Direct:
    os << "direct(" << value << ")";
    return;
  case Kind::InstResult:
    os << "instResult(" << cast<InstanceOp>(op).getName() << ", " << index
       << ")";
    return;
  case Kind::ModuleArg:
    os << "moduleArg(" << cast<FModuleOp>(op).getModuleName() << ", " << index
       << ")";
    return;
  case Kind::PlannedWire:
    os << "plannedWire(" << index << ")";
    return;
  case Kind::GateEnable:
    os << "gateEnable(" << *op << ")";
    return;
  }
}

Value GatedClockConversion::resolve(MatRef ref) {
  switch (ref.getKind()) {
  case MatRef::Kind::None:
    return Value();
  case MatRef::Kind::Direct:
    return ref.getValue();
  case MatRef::Kind::InstResult:
    return liveInstance(ref.getOp())->getResult(ref.getIndex());
  case MatRef::Kind::ModuleArg:
    return cast<FModuleOp>(ref.getOp())
        .getBodyBlock()
        ->getArgument(ref.getIndex());
  case MatRef::Kind::PlannedWire:
    return plannedWireValues[ref.getIndex()];
  case MatRef::Kind::GateEnable:
    return gateEnableOf(ref.gate());
  }
  llvm_unreachable("unhandled MatRef kind");
}

unsigned GatedClockConversion::newEnableNode(unsigned parent, MatRef term,
                                             Location loc, MatRef anchor) {
  enableNodes.push_back({parent, term, anchor, loc});
  return enableNodes.size() - 1;
}

Value GatedClockConversion::lower(unsigned enableId) {
  if (enableId == kNoEnable)
    return Value();
  loweredEnables.resize(enableNodes.size());
  if (Value cached = loweredEnables[enableId])
    return cached;

  // Walk up to the first already-lowered node, then emit from there back down.
  // Iterative so that a long gate cascade cannot overflow the stack.
  SmallVector<unsigned> chain;
  unsigned cur = enableId;
  while (cur != kNoEnable && !loweredEnables[cur]) {
    chain.push_back(cur);
    cur = enableNodes[cur].parent;
  }

  Value upstream = cur == kNoEnable ? Value() : loweredEnables[cur];
  for (unsigned id : llvm::reverse(chain)) {
    const EnableNode &node = enableNodes[id];
    Value result = resolve(node.term);
    if (node.parent != kNoEnable) {
      // AND with the upstream enable, so the register holds whenever any gate
      // in the chain is closed. The anchor is the clock this enable
      // accompanies, which dominates every consumer of the pair.
      assert(upstream && "an upstream enable must lower to a value");
      ImplicitLocOpBuilder builder(node.loc, context);
      builder.setInsertionPointAfterValue(resolve(node.anchor));
      result = builder.createOrFold<AndPrimOp>(upstream, result);
    }
    loweredEnables[id] = result;
    upstream = result;
  }
  return upstream;
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: value materialization helpers
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

  // At the top of the body, so it dominates every possible use.
  ImplicitLocOpBuilder builder(mod.getLoc(), context);
  builder.setInsertionPointToStart(mod.getBodyBlock());
  Value constOne = builder.createOrFold<ConstantOp>(
      APSInt(APInt(1, 1, /*isSigned=*/false), /*isUnsigned=*/true));
  constU1Cache[mod] = constOne;
  return constOne;
}

void GatedClockConversion::connectMaterializedToInstancePorts(
    InstanceOp inst, unsigned clkPortIndex, unsigned enPortIndex,
    Value materializedClk, Value materializedEn) {
  ImplicitLocOpBuilder builder(inst.getLoc(), context);
  // At the end of the block, so the materialized clock dominates the connect.
  builder.setInsertionPointToEnd(inst->getBlock());

  MatchingConnectOp::create(builder, inst->getResult(clkPortIndex),
                            materializedClk);

  if (!materializedEn)
    materializedEn = getOrCreateConstU1One(inst->getParentOfType<FModuleOp>());
  MatchingConnectOp::create(builder, inst->getResult(enPortIndex),
                            materializedEn);
}

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

//===----------------------------------------------------------------------===//
// GatedClockConversion: root rewriting
//===----------------------------------------------------------------------===//

LogicalResult GatedClockConversion::rewriteRoot(Operation *op, Value baseClk,
                                                Value enable) {
  if (!enable)
    return success();

  // RefForce/RefRelease: rebind the clock to the ungated base and fold the
  // enable into the predicate.
  if (auto fop = dyn_cast<RefForceOp>(op)) {
    fop.getClockMutable().assign(baseClk);
    ImplicitLocOpBuilder b(fop.getLoc(), fop);
    fop.getPredicateMutable().assign(
        b.createOrFold<AndPrimOp>(fop.getPredicate(), enable));
    return success();
  }
  if (auto rop = dyn_cast<RefReleaseOp>(op)) {
    rop.getClockMutable().assign(baseClk);
    ImplicitLocOpBuilder b(rop.getLoc(), rop);
    rop.getPredicateMutable().assign(
        b.createOrFold<AndPrimOp>(rop.getPredicate(), enable));
    return success();
  }

  Value regData;
  if (auto reg = dyn_cast<RegOp>(op))
    regData = reg.getData();
  else if (auto regr = dyn_cast<RegResetOp>(op))
    regData = regr.getData();
  else
    return op->emitError("unsupported for gated clock conversion");

  // ExpandWhens leaves exactly one write per register. Rebinding the clock
  // without sinking the enable would silently drop the gate, so bail out if
  // that precondition does not hold.
  FConnectLike dataWrite;
  unsigned writers = 0;
  for (auto &use : regData.getUses()) {
    auto fconn = dyn_cast<FConnectLike>(use.getOwner());
    if (fconn && fconn.getDest() == regData) {
      ++writers;
      dataWrite = fconn;
    }
  }
  if (writers != 1) {
    op->emitWarning() << "gated clock conversion: expected exactly one connect "
                         "driving this register (run after "
                         "firrtl-expand-whens); found "
                      << writers << "; leaving the gated clock in place";
    return success();
  }

  // Rebind to the ungated base and wrap the write with mux(enable, RHS,
  // regData), so the register holds while the clock gate is closed.
  op->setOperand(0, baseClk);
  ImplicitLocOpBuilder b(dataWrite.getLoc(), dataWrite);
  Value newRhs = b.createOrFold<MuxPrimOp>(enable, dataWrite.getSrc(), regData);
  dataWrite->setOperand(1, newRhs);
  return success();
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: planning (no IR mutation)
//===----------------------------------------------------------------------===//

void GatedClockConversion::planAlias(Value dstClk, FModuleOp srcMod,
                                     MatRef baseClk, unsigned enableId) {
  if (enableId == kNoEnable) {
    clockEnablePairs[dstClk] = {baseClk, kNoEnable};
    return;
  }
  // A wire pair carries (base, enable) past the alias;
  // eliminateTemporaryWires() forwards it away when safe.
  unsigned wireId = 2 * wirePlans.size();
  wirePlans.push_back({srcMod, baseClk, enableId, dstClk.getLoc()});
  clockEnablePairs[dstClk] = {
      MatRef::plannedWire(wireId),
      newEnableLeaf(MatRef::plannedWire(wireId + 1), dstClk.getLoc())};
}

void GatedClockConversion::planGate(ClockGateIntrinsicOp gate, Value dstClk,
                                    MatRef baseClk, unsigned enableId) {
  // The base clock passes through unchanged, so cascaded gates all resolve to
  // the same ungated base.
  auto gateEn = newEnableNode(enableId, MatRef::gateEnable(gate), gate.getLoc(),
                              /*anchor=*/MatRef::of(dstClk));
  clockEnablePairs[dstClk] = {baseClk, gateEn};
}

void GatedClockConversion::recordInstanceDrive(InstanceOp inst,
                                               const PortPairPlan &plan,
                                               MatRef baseClk,
                                               unsigned enableId) {
  assert(plan.dir == Direction::In &&
         "only input port pairs are driven at the caller");
  // A clock loop closed through an instance (`inst.clk_in <- inst.clk_out`) is
  // deliberately not diagnosed here: doing it precisely needs an
  // instance-path-sensitive analysis, and `firrtl-check-comb-loops` already
  // rejects such input.

  // Keyed by (instance, port index), so a second edge reaching an already
  // planned pair from the same caller is a no-op, not a duplicate connect.
  instanceDrives.try_emplace(
      {inst, plan.baseIdx},
      InstanceDrive{inst, plan.baseIdx, plan.enIdx, baseClk, enableId});
}

std::pair<unsigned, unsigned>
GatedClockConversion::planGatedPorts(InstanceOp inst, FModuleOp childMod,
                                     unsigned gatedClkIndex, Direction dir,
                                     MatRef baseClk, unsigned enableId) {
  PortPlanKey key{childMod, gatedClkIndex};
  auto *it = portPlans.find(key);
  if (it == portPlans.end()) {
    // The final indices are known already: `insertPlannedPorts()` only appends
    // ports, so existing indices never shift (asserted when applying).
    unsigned &nextIdx =
        nextPortIdx.try_emplace(childMod, childMod.getNumPorts()).first->second;
    PortPairPlan plan({childMod, gatedClkIndex, dir, nextIdx, nextIdx + 1});
    nextIdx += 2;
    if (dir == Direction::Out) {
      assert(enableId != kNoEnable &&
             "unless this is a gated clock, no need to add output enable port");
      plan.outBaseClk = baseClk;
      plan.outEnableId = enableId;
    }
    it = portPlans.insert({key, plan}).first;
    plansPerModule[childMod].push_back(key);
  }

  const PortPairPlan &plan = it->second;
  assert(plan.dir == dir && "a port cannot change direction");
  // Every caller of the module must drive a planned input pair.
  if (dir == Direction::In)
    recordInstanceDrive(inst, plan, baseClk, enableId);
  return {plan.baseIdx, plan.enIdx};
}

void GatedClockConversion::planInstancePort(Direction dir, InstanceOp inst,
                                            Value dstClk, Value srcClk,
                                            MatRef baseClk, unsigned enableId) {
  auto childMod =
      dyn_cast_or_null<FModuleOp>(inst.getReferencedModule(ig).getOperation());
  auto gatedClkIndex = cast<OpResult>(srcClk).getResultNumber();
  if (enableId == kNoEnable) {
    if (dir == Direction::Out) {
      // Symbolic, because a port pair for a *different* clock port of this
      // module would clone all of its instances.
      clockEnablePairs[dstClk] = {MatRef::of(dstClk), kNoEnable};
      return;
    }
    // This instance drives an ungated clock input, but a sibling instance may
    // drive a gated one, in which case the pair is added for *every* instance.
    // `gatedClocks` answers that without blocking on the sibling's pair, which
    // in a same-module cascade would depend on this very port.
    if (!gatedClocks.contains(dstClk)) {
      clockEnablePairs[dstClk] = {MatRef::of(dstClk), kNoEnable};
      return;
    }

    // A sibling is gated: add the pair here too. `kNoEnable` on the resulting
    // instance drive connects a constant 1.
  }
  assert((enableId == kNoEnable || dir == Direction::Out ||
          gatedClocks.contains(dstClk)) &&
         "a gated pair must imply a gated mark");
  auto [baseClkIndex, enableIndex] =
      planGatedPorts(inst, childMod, gatedClkIndex, dir, baseClk, enableId);

  // An output pair is read on the instance result side, an input pair on the
  // child's block-argument side. Neither exists yet, hence symbolic refs.
  MatRef baseRef, enRef;
  if (dir == Direction::Out) {
    baseRef = MatRef::instResult(inst, baseClkIndex);
    enRef = MatRef::instResult(inst, enableIndex);
  } else {
    baseRef = MatRef::moduleArg(childMod, baseClkIndex);
    enRef = MatRef::moduleArg(childMod, enableIndex);
  }
  assert((dir == Direction::Out ? inst->getParentOfType<FModuleOp>()
                                : childMod) == getParentModule(dstClk) &&
         "parent modules must match");
  clockEnablePairs[dstClk] = {baseRef, newEnableLeaf(enRef, dstClk.getLoc())};
}

void GatedClockConversion::planMultiplyInstantiatedInput(Value srcClk,
                                                         MatRef baseClk,
                                                         unsigned enableId) {
  // `srcClk` is a result of the caller instance; drive the port pair that an
  // earlier caller of this module already planned.
  auto inst = srcClk.getDefiningOp<InstanceOp>();
  assert(inst);
  auto childMod =
      dyn_cast_or_null<FModuleOp>(inst.getReferencedModule(ig).getOperation());
  auto gatedClkIndex = cast<OpResult>(srcClk).getResultNumber();
  auto *it = portPlans.find({childMod, gatedClkIndex});
  // No plan means an output port, which the InstanceOut path handles.
  if (it == portPlans.end()) {
    LLVM_DEBUG(llvm::dbgs() << "  no plan for index " << gatedClkIndex
                            << ", skipping drive (handled by InstanceOut)\n");
    return;
  }
  recordInstanceDrive(inst, it->second, baseClk, enableId);
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

Value GatedClockConversion::processEdge(const ClockEdge &edge, Value srcClk,
                                        FModuleOp srcMod, MatRef baseClk,
                                        unsigned enableId) {
  LLVM_DEBUG(llvm::dbgs() << "  edge kind=" << edgeKindName(edge.kind) << "\n");

  if (clockEnablePairs.count(edge.dst)) {
    // For InstanceIn this is a multiply-instantiated module whose ports an
    // earlier caller planned; this caller still has to drive them.
    if (edge.kind == EdgeKind::InstanceIn)
      planMultiplyInstantiatedInput(srcClk, baseClk, enableId);
    return Value();
  }

  switch (edge.kind) {
  case EdgeKind::Alias:
    planAlias(edge.dst, srcMod, baseClk, enableId);
    break;
  case EdgeKind::Gate:
    planGate(edge.gate(), edge.dst, baseClk, enableId);
    break;
  case EdgeKind::InstanceIn:
    planInstancePort(Direction::In, edge.instance(), edge.dst, srcClk, baseClk,
                     enableId);
    break;
  case EdgeKind::InstanceOut:
    planInstancePort(Direction::Out, edge.instance(), edge.dst, edge.dst,
                     baseClk, enableId);
    break;
  }
  // `edge.dst` now has a pair, so it is safe to visit next.
  assert(clockEnablePairs.count(edge.dst) && "the destination must be planned");
  return edge.dst;
}

void GatedClockConversion::plan() {
  LLVM_DEBUG(llvm::dbgs() << "[plan] " << baseClks.size() << " base clocks\n");
  // Propagate (base, enable) pairs from the base clocks through the clock flow
  // graph, planning ports as needed.
  //
  // This terminates on any graph, cyclic or not: a node is enqueued only once
  // it has a pair and `processEdge` skips destinations that already have one.
  // Clocks in a loop with no base clock are never planned; `run()` reports
  // them.
  //
  // BFS rather than DFS purely to keep the emission order stable.
  std::deque<Value> worklist(baseClks.begin(), baseClks.end());
  for (auto baseClk : baseClks)
    clockEnablePairs[baseClk] = {MatRef::of(baseClk), kNoEnable};

  while (!worklist.empty()) {
    auto srcClk = worklist.front();
    worklist.pop_front();
    FModuleOp srcMod = getParentModule(srcClk);

    auto it = clockEnablePairs.find(srcClk);
    assert(it != clockEnablePairs.end() &&
           "a node is only enqueued once it has a pair");
    // Copy the pair out: `processEdge` inserts into `clockEnablePairs` and
    // invalidates `it`.
    MatRef baseClk = it->second.baseClk;
    unsigned enableId = it->second.enableId;

    for (auto &edge : srcToDstClocks[srcClk])
      if (Value next = processEdge(edge, srcClk, srcMod, baseClk, enableId))
        worklist.push_back(next);
  }
  LLVM_DEBUG(llvm::dbgs() << "[plan] complete\n");
}

//===----------------------------------------------------------------------===//
// GatedClockConversion: applyPlan (the only IR-mutating phase)
//===----------------------------------------------------------------------===//

void GatedClockConversion::createPlannedWires() {
  auto createWire = [&](Type type, ImplicitLocOpBuilder &builder) {
    auto w = WireOp::create(builder, type);
    wireOps.push_back(w);
    return w.getData();
  };
  plannedWireValues.reserve(2 * wirePlans.size());
  for (auto &wirePlan : wirePlans) {
    // Wires have no operands, so they can all be created up front.
    auto builder = ImplicitLocOpBuilder::atBlockBegin(
        wirePlan.loc, wirePlan.mod.getBodyBlock());
    plannedWireValues.push_back(createWire(clockType, builder));
    plannedWireValues.push_back(createWire(u1Type, builder));
  }
}

void GatedClockConversion::insertPlannedPorts() {
  for (auto &[mod, keys] : plansPerModule) {
    // All pairs of a module are appended in one call, so every instance is
    // re-created exactly once no matter how many pairs the module needs.
    const unsigned origNumPorts = mod.getNumPorts();
    SmallVector<std::pair<unsigned, PortInfo>> newPorts;
    for (auto key : keys) {
      const PortPairPlan &portPlan = portPlans.find(key)->second;
      assert(portPlan.baseIdx == origNumPorts + newPorts.size() &&
             "port index pre-assignment invalidated: ports were inserted "
             "outside applyPlan()");
      auto [baseInfo, enableInfo] = makeGatedClockPortInfos(
          context, mod.getPortName(portPlan.gatedClkIndex), portPlan.dir,
          mod.getLoc(), clockType, u1Type);
      newPorts.emplace_back(origNumPorts, baseInfo);
      newPorts.emplace_back(origNumPorts, enableInfo);
    }
    mod.insertPorts(newPorts);

    // A result list cannot grow in place, so every instance has to be
    // re-created. Collect them first: cloning updates the use list.
    auto *node = ig.lookup(mod);
    SmallVector<InstanceOp> oldInsts;
    for (auto *use : node->uses())
      if (auto i = dyn_cast<InstanceOp>(*use->getInstance()))
        oldInsts.push_back(i);

    for (auto oldInst : oldInsts) {
      auto cloneIface = oldInst.cloneWithInsertedPortsAndReplaceUses(newPorts);
      auto newInst = cast<InstanceOp>(cloneIface.getOperation());
      ig.replaceInstance(oldInst, newInst);
      assert(!instClones.count(oldInst) && "instance re-created twice");
      instClones[oldInst] = newInst;
      // Defer erasure until nothing reads the plan any more.
      deadInstances.push_back(oldInst);
    }
  }
}

LogicalResult GatedClockConversion::emitPlannedIR() {
  // Planned output port pairs: drive the new ports from inside the module.
  for (auto &[key, portPlan] : portPlans) {
    if (portPlan.dir != Direction::Out)
      continue;
    Value materializedClk = resolve(portPlan.outBaseClk);
    Value materializedEn = lower(portPlan.outEnableId);
    auto *body = portPlan.mod.getBodyBlock();
    ImplicitLocOpBuilder builder(portPlan.mod.getLoc(), context);
    builder.setInsertionPointToEnd(body);
    MatchingConnectOp::create(builder, body->getArgument(portPlan.baseIdx),
                              materializedClk);
    MatchingConnectOp::create(builder, body->getArgument(portPlan.enIdx),
                              materializedEn);
  }

  // Planned input port pairs: drive them at every caller instance.
  for (auto &[key, drive] : instanceDrives) {
    Value materializedClk = resolve(drive.baseClk);
    Value materializedEn = lower(drive.enableId);
    connectMaterializedToInstancePorts(
        cast<InstanceOp>(liveInstance(drive.inst)), drive.baseIdx, drive.enIdx,
        materializedClk, materializedEn);
  }

  // Planned carrier wires: connect them to their source pair.
  for (auto [index, wirePlan] : llvm::enumerate(wirePlans)) {
    Value clockWire = plannedWireValues[2 * index];
    Value enWire = plannedWireValues[2 * index + 1];
    Value materializedClk = resolve(wirePlan.baseClk);
    Value materializedEn = lower(wirePlan.enableId);
    ImplicitLocOpBuilder builder(wirePlan.loc, context);
    builder.setInsertionPointAfter(enWire.getDefiningOp());
    if (!isa<BlockArgument>(materializedClk))
      builder.setInsertionPointAfterValue(materializedClk);
    MatchingConnectOp::create(builder, clockWire, materializedClk);
    if (!isa<BlockArgument>(materializedEn))
      builder.setInsertionPointAfterValue(materializedEn);
    MatchingConnectOp::create(builder, enWire, materializedEn);
  }

  // Root rewrites, last: they consume the fully materialized pairs.
  for (const auto &rewrite : rootRewrites)
    if (failed(rewriteRoot(rewrite.op, resolve(rewrite.baseClk),
                           lower(rewrite.enableId))))
      return failure();
  return success();
}

LogicalResult GatedClockConversion::applyPlan() {
  createPlannedWires();
  insertPlannedPorts();
  return emitPlannedIR();
}

void GatedClockConversion::eliminateTemporaryWires() {
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
      writeConnect.erase();
      wire.erase();
    }
  }
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

void GatedClockConversion::dumpPlan() const {
  auto &os = llvm::dbgs();
  auto printEnable = [&](unsigned id) {
    if (id == kNoEnable) {
      os << "<none>";
      return;
    }
    // Print the accumulation chain leaf-to-root, i.e. as it will be ANDed.
    for (unsigned cur = id; cur != kNoEnable; cur = enableNodes[cur].parent) {
      if (cur != id)
        os << " & ";
      enableNodes[cur].term.print(os);
    }
  };

  os << "=== Planned wire pairs ===\n";
  for (auto [index, wirePlan] : llvm::enumerate(wirePlans)) {
    FModuleOp mod = wirePlan.mod;
    os << "  #" << 2 * index << "/" << 2 * index + 1 << " in "
       << mod.getModuleName() << " <- ";
    wirePlan.baseClk.print(os);
    os << ", ";
    printEnable(wirePlan.enableId);
    os << "\n";
  }
  os << "=== Planned port pairs ===\n";
  for (const auto &[key, portPlan] : portPlans) {
    FModuleOp mod = portPlan.mod;
    os << "  " << mod.getModuleName() << "."
       << mod.getPortName(portPlan.gatedClkIndex) << ": "
       << (portPlan.dir == Direction::In ? "in" : "out") << " @"
       << portPlan.baseIdx << "/" << portPlan.enIdx;
    if (portPlan.dir == Direction::Out) {
      os << " <- ";
      portPlan.outBaseClk.print(os);
      os << ", ";
      printEnable(portPlan.outEnableId);
    }
    os << "\n";
  }
  os << "=== Planned instance drives ===\n";
  for (const auto &[key, drive] : instanceDrives) {
    InstanceOp inst = drive.inst;
    os << "  " << inst.getName() << " @" << drive.baseIdx << "/" << drive.enIdx
       << " <- ";
    drive.baseClk.print(os);
    os << ", ";
    printEnable(drive.enableId);
    os << "\n";
  }
  os << "=== Planned root rewrites ===\n";
  for (const auto &rewrite : rootRewrites) {
    os << "  " << rewrite.op->getName() << " <- ";
    rewrite.baseClk.print(os);
    os << ", ";
    printEnable(rewrite.enableId);
    os << "\n";
  }
  os << "======================\n";
}
