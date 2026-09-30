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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

#define DEBUG_TYPE "firrtl-gated-clock-conversion"

using namespace circt;
using namespace firrtl;

namespace {

StringRef edgeKindName(EdgeKind kind) {
  switch (kind) {
  case EdgeKind::Gate:
    return "Gate";
  case EdgeKind::InstanceIn:
    return "InstanceIn";
  case EdgeKind::InstanceOut:
    return "InstanceOut";
  }
  return "?";
}

std::pair<PortInfo, PortInfo>
makeGatedClockPortInfos(MLIRContext *ctx, StringRef tag, Direction dir,
                        Location loc, Type clockType, Type u1Type) {
  return {PortInfo(StringAttr::get(ctx, ("_gatedClock_baseClock_" + tag).str()),
                   clockType, dir, /*symName=*/StringAttr(), loc),
          PortInfo(StringAttr::get(ctx, ("_gatedClock_enable_" + tag).str()),
                   u1Type, dir, /*symName=*/StringAttr(), loc)};
}

FModuleOp getParentModule(Value value) {
  if (isa<BlockArgument>(value))
    return cast<FModuleOp>(value.getParentBlock()->getParentOp());
  return value.getDefiningOp()->getParentOfType<FModuleOp>();
}

/// The `*_initial` force/release variants have no clock, so they are not
/// roots.
Value clockOperandOf(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<RefForceOp, RefReleaseOp>([](auto op) { return op.getClock(); })
      .Case<RegOp, RegResetOp>([](auto op) { return op.getClockVal(); })
      .Case<ClockGateIntrinsicOp>([](auto op) { return op.getInput(); })
      .Default([](auto) { return Value(); });
}

/// The driver of `clk` through wires, nodes and casts; null if undriven.
Value clockDriver(Value clk) {
  Value driver = getModuleScopedDriver(clk, /*lookThroughWires=*/true,
                                       /*lookThroughNodes=*/true,
                                       /*lookThroughCasts=*/true);
  // `asClock(asUInt(clk))` is `clk`, but `asClock(u)` is a clock of its own.
  if (driver && !type_isa<ClockType>(driver.getType()))
    driver = getModuleScopedDriver(clk, /*lookThroughWires=*/true,
                                   /*lookThroughNodes=*/true,
                                   /*lookThroughCasts=*/false);
  return driver;
}

/// A driver-normal clock that is neither a gate nor passed in through a port.
bool isBaseClock(Value clk, InstanceGraph &ig) {
  if (clockDriver(clk) != clk)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(clk))
    return ig.lookup(getParentModule(arg))->noUses();
  Operation *op = clk.getDefiningOp();
  if (isa<ClockGateIntrinsicOp>(op))
    return false;
  if (auto inst = dyn_cast<InstanceOp>(op))
    return !isa<FModuleOp>(inst.getReferencedModule(ig).getOperation());
  return true;
}

/// Whichever of `a` and `b` an op must be inserted after to see both.
Value laterOf(Value a, Value b) {
  Operation *defA = a.getDefiningOp(), *defB = b.getDefiningOp();
  if (!defA)
    return b;
  if (!defB)
    return a;
  Operation *ancestor = defA->getBlock()->findAncestorOpInBlock(*defB);
  if (!ancestor)
    return a;
  return ancestor == defA || defA->isBeforeInBlock(ancestor) ? b : a;
}

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

Value GatedClockConversion::live(Value v) const {
  if (auto result = dyn_cast<OpResult>(v))
    if (auto *clone = instClones.lookup(result.getOwner()))
      return clone->getResult(result.getResultNumber());
  return v;
}

Value GatedClockConversion::gateEnableOf(ClockGateIntrinsicOp gate) {
  if (!gate.getTestEnable())
    return gate.getEnable();
  ImplicitLocOpBuilder b(gate.getLoc(), gate);
  return b.createOrFold<OrPrimOp>(gate.getEnable(), gate.getTestEnable());
}

Value GatedClockConversion::andEnables(Value upstream, Value gateEn,
                                       Location loc) {
  if (!upstream)
    return gateEn;
  ImplicitLocOpBuilder builder(loc, context);
  builder.setInsertionPointAfterValue(laterOf(upstream, gateEn));
  return builder.createOrFold<AndPrimOp>(upstream, gateEn);
}

Value GatedClockConversion::getOrCreateConstU1One(FModuleOp mod) {
  auto it = constU1Cache.find(mod);
  if (it != constU1Cache.end())
    return it->second;

  // At the top of the body, so it dominates every use.
  ImplicitLocOpBuilder builder(mod.getLoc(), context);
  builder.setInsertionPointToStart(mod.getBodyBlock());
  Value constOne = builder.createOrFold<ConstantOp>(
      APSInt(APInt(1, 1, /*isSigned=*/false), /*isUnsigned=*/true));
  constU1Cache[mod] = constOne;
  return constOne;
}

void GatedClockConversion::drivePair(Location loc, Value dstClk, Value dstEn,
                                     Value clk, Value en) {
  // At the end of the block, so `clk` and `en` dominate the connects.
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, dstClk.getParentBlock());
  MatchingConnectOp::create(builder, dstClk, clk);
  if (!en)
    en = getOrCreateConstU1One(getParentModule(dstClk));
  MatchingConnectOp::create(builder, dstEn, en);
}

Value GatedClockConversion::getDominatingValue(Operation *user, Value v) {
  auto *def = v.getDefiningOp();
  if (!def)
    return v;
  Operation *ancestor = def->getBlock()->findAncestorOpInBlock(*user);
  assert(ancestor && "a value is used within the block that defines it");
  if (def->isBeforeInBlock(ancestor))
    return v;

  // `user` reads `v` through a wire driven later in the block.
  auto [it, inserted] = dominatingValues.try_emplace(v);
  if (inserted) {
    ImplicitLocOpBuilder builder(v.getLoc(), context);
    builder.setInsertionPointToStart(def->getBlock());
    it->second = WireOp::create(builder, v.getType()).getData();
    builder.setInsertionPointAfterValue(v);
    MatchingConnectOp::create(builder, it->second, v);
  }
  return it->second;
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

LogicalResult GatedClockConversion::addRoot(Operation *op) {
  aliasesBuilt = false;
  Value clk = clockOperandOf(op);
  if (!clk)
    return op->emitError(
        "unsupported operation type for gated clock "
        "conversion; expected RefForceOp, RefReleaseOp, RegOp, "
        "RegResetOp or ClockGateIntrinsicOp");
  Root root{op, clk};

  // The enable is sunk into the register's write, so there must be exactly one.
  Value regData =
      TypeSwitch<Operation *, Value>(op)
          .Case<RegOp, RegResetOp>([](auto reg) { return reg.getData(); })
          .Default([](auto) { return Value(); });
  if (regData) {
    for (auto &use : regData.getUses()) {
      auto fconn = dyn_cast<FConnectLike>(use.getOwner());
      if (fconn && fconn.getDest() == regData) {
        ++root.numWrites;
        root.dataWrite = fconn;
      }
    }
    if (root.numWrites != 1)
      root.dataWrite = {};
  }
  roots.push_back(root);
  return success();
}

LogicalResult GatedClockConversion::analyze() {
  LogicalResult result = success();
  SmallVector<Value> worklist;
  bool needed = true;

  // Without a driver there is no pair to give the clock, and a caller of an
  // input port pair would be left not driving it.
  auto reportUndriven = [&](Value clk) {
    mlir::emitError(clk.getLoc())
        << "gated clock conversion: this clock is not driven; run this "
           "utility after firrtl-expand-whens and firrtl-check-init";
    result = failure();
  };

  auto enqueue = [&](Value clk) {
    if (analyzed.try_emplace(clk, needed).second)
      worklist.push_back(clk);
  };

  auto addEdge = [&](Value dstClk, Value srcClk, Operation *op, EdgeKind kind) {
    Value driver = clockDriver(srcClk);
    if (!driver)
      return reportUndriven(srcClk);
    LLVM_DEBUG(llvm::dbgs() << "  edge kind=" << edgeKindName(kind) << "\n");
    srcToDstClocks[driver].push_back({dstClk, op, kind});
    enqueue(driver);
  };

  auto visit = [&](Value clk) {
    if (auto blockArg = dyn_cast<BlockArgument>(clk)) {
      auto mod = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
      unsigned portIdx = blockArg.getArgNumber();
      auto *node = ig.lookup(mod);
      if (node->noUses()) {
        baseClks.push_back(clk);
        return;
      }
      for (auto *use : node->uses()) {
        if (auto callerInst = dyn_cast<InstanceOp>(*use->getInstance())) {
          addEdge(clk, callerInst.getResult(portIdx), callerInst,
                  EdgeKind::InstanceIn);
        } else {
          use->getInstance()->emitError("can only handle InstanceOp");
          result = failure();
        }
      }
      return;
    }
    auto *defOp = clk.getDefiningOp();

    if (auto gate = dyn_cast<ClockGateIntrinsicOp>(defOp))
      return addEdge(clk, gate.getInput(), gate, EdgeKind::Gate);

    if (auto inst = dyn_cast<InstanceOp>(defOp)) {
      auto childMod = dyn_cast_or_null<FModuleOp>(
          inst.getReferencedModule(ig).getOperation());
      if (!childMod) {
        baseClks.push_back(clk);
        return;
      }
      unsigned portIdx = cast<OpResult>(clk).getResultNumber();
      return addEdge(clk, childMod.getBodyBlock()->getArgument(portIdx), inst,
                     EdgeKind::InstanceOut);
    }

    // A mux of gated clocks has no single enable to sink. Stay silent for
    // ordinary clock selection, which leaves nothing behind.
    if (auto mux = dyn_cast<MuxPrimOp>(defOp)) {
      Value inputs[] = {mux.getHigh(), mux.getLow()};
      if (llvm::any_of(inputs, [](Value v) {
            Value d = clockDriver(v);
            return d && d.getDefiningOp<ClockGateIntrinsicOp>();
          }))
        mlir::emitRemark(mux.getLoc())
            << "gated clock conversion: clock selection is not supported; the "
               "clock gate feeding this mux was left in place";
    }

    baseClks.push_back(clk);
  };

  // Unsinkable roots go last, and only to tell whether they are gated, so
  // that no port pair is added for them alone.
  for (bool sinkable : {true, false}) {
    needed = sinkable;
    for (Root &root : roots) {
      if (root.isSinkable() != sinkable)
        continue;
      root.key = clockDriver(root.clock);
      if (!root.key) {
        reportUndriven(root.clock);
        continue;
      }
      enqueue(root.key);
    }
    while (!worklist.empty())
      visit(worklist.pop_back_val());
  }
  return result;
}

void GatedClockConversion::markGated() {
  // Requeue a clock only when its bit flips, so each is visited at most twice.
  SmallVector<Value> worklist;
  for (Value baseClk : baseClks) {
    gated.try_emplace(baseClk, false);
    worklist.push_back(baseClk);
  }
  while (!worklist.empty()) {
    Value src = worklist.pop_back_val();
    auto edges = srcToDstClocks.find(src);
    if (edges == srcToDstClocks.end())
      continue;
    bool srcGated = gated.lookup(src);
    for (const ClockEdge &edge : edges->second) {
      bool dstGated = srcGated || edge.kind == EdgeKind::Gate;
      auto [it, inserted] = gated.try_emplace(edge.dst, dstGated);
      if (!inserted) {
        if (!dstGated || it->second)
          continue;
        it->second = true;
      }
      worklist.push_back(edge.dst);
    }
  }
}

LogicalResult GatedClockConversion::checkRoots() {
  LogicalResult result = success();
  for (const Root &root : roots) {
    auto it = gated.find(root.key);
    if (it == gated.end()) {
      mlir::emitWarning(root.clock.getLoc())
          << "gated clock conversion: this clock is not reachable from any "
             "free-running base clock (clock feedback loop?); leaving the op "
             "unchanged";
      continue;
    }
    if (!it->second)
      continue;
    if (isa<ClockGateIntrinsicOp>(root.op)) {
      root.op->emitError("unsupported for gated clock conversion");
      result = failure();
      continue;
    }
    // Rebinding the clock without sinking the enable would drop the gate.
    if (!root.isSinkable())
      root.op->emitWarning()
          << "gated clock conversion: expected exactly one connect driving "
             "this register (run after firrtl-expand-whens); found "
          << root.numWrites << "; leaving the gated clock in place";
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Mutation
//===----------------------------------------------------------------------===//

SmallVector<Value> GatedClockConversion::selectPortPairs() {
  // BFS, which `materialize()` relies on.
  SmallVector<Value> order;
  DenseSet<Value> seen;
  auto push = [&](Value clk) {
    if (analyzed.lookup(clk) && seen.insert(clk).second)
      order.push_back(clk);
  };
  for (Value baseClk : baseClks)
    push(baseClk);

  for (size_t i = 0; i < order.size(); ++i) {
    Value src = order[i];
    auto edges = srcToDstClocks.find(src);
    if (edges == srcToDstClocks.end())
      continue;
    for (const ClockEdge &edge : edges->second) {
      if (!analyzed.lookup(edge.dst))
        continue;
      if (edge.kind == EdgeKind::InstanceIn && gated.lookup(edge.dst)) {
        auto arg = cast<BlockArgument>(edge.dst);
        portPairs.try_emplace({cast<FModuleOp>(arg.getOwner()->getParentOp()),
                               arg.getArgNumber()},
                              PortPair{Direction::In});
      } else if (edge.kind == EdgeKind::InstanceOut && gated.lookup(src)) {
        portPairs.try_emplace(
            {getParentModule(src), cast<OpResult>(edge.dst).getResultNumber()},
            PortPair{Direction::Out});
      }
      push(edge.dst);
    }
  }
  return order;
}

void GatedClockConversion::insertPorts() {
  llvm::MapVector<FModuleOp, SmallVector<std::pair<unsigned, PortPair *>>>
      perModule;
  for (auto &[key, portPair] : portPairs)
    perModule[key.first].push_back({key.second, &portPair});

  for (auto &[mod, modPairs] : perModule) {
    // One call per module, so each instance is re-created once.
    llvm::sort(modPairs, llvm::less_first());
    const unsigned origNumPorts = mod.getNumPorts();
    SmallVector<std::pair<unsigned, PortInfo>> newPorts;
    for (auto [portIdx, portPair] : modPairs) {
      portPair->baseIdx = origNumPorts + newPorts.size();
      portPair->enIdx = portPair->baseIdx + 1;
      auto [baseInfo, enableInfo] = makeGatedClockPortInfos(
          context, mod.getPortName(portIdx), portPair->dir, mod.getLoc(),
          clockType, u1Type);
      newPorts.emplace_back(origNumPorts, baseInfo);
      newPorts.emplace_back(origNumPorts, enableInfo);
    }
    mod.insertPorts(newPorts);

    // Collect first: cloning updates the use list.
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
      deadInstances.push_back(oldInst);
    }
  }
}

void GatedClockConversion::materialize(ArrayRef<Value> order) {
  for (Value baseClk : baseClks)
    if (analyzed.lookup(baseClk))
      pairs[baseClk] = {live(baseClk), Value()};

  for (Value src : order) {
    auto [baseClk, enable] = pairs.at(src);
    assert(!enable == !gated.lookup(src) && "a clock has an enable iff gated");
    auto edges = srcToDstClocks.find(src);
    if (edges == srcToDstClocks.end())
      continue;
    for (const ClockEdge &edge : edges->second) {
      if (!analyzed.lookup(edge.dst))
        continue;
      switch (edge.kind) {
      case EdgeKind::Gate: {
        auto gate = edge.gate();
        [[maybe_unused]] bool inserted =
            pairs
                .try_emplace(
                    edge.dst, baseClk,
                    andEnables(enable, gateEnableOf(gate), gate.getLoc()))
                .second;
        assert(inserted && "a gate is reached by a single edge");
        break;
      }
      case EdgeKind::InstanceIn: {
        auto arg = cast<BlockArgument>(edge.dst);
        auto childMod = cast<FModuleOp>(arg.getOwner()->getParentOp());
        auto *portPair = portPairs.find({childMod, arg.getArgNumber()});
        if (portPair == portPairs.end()) {
          pairs.try_emplace(arg, arg, Value());
          break;
        }
        // Every caller drives the pair, so this is done on every edge.
        const PortPair &pp = portPair->second;
        auto *inst = liveInstance(edge.op);
        drivePair(inst->getLoc(), inst->getResult(pp.baseIdx),
                  inst->getResult(pp.enIdx), baseClk, enable);
        Block *body = childMod.getBodyBlock();
        pairs.try_emplace(arg, body->getArgument(pp.baseIdx),
                          body->getArgument(pp.enIdx));
        break;
      }
      case EdgeKind::InstanceOut: {
        auto *portPair = portPairs.find(
            {getParentModule(src), cast<OpResult>(edge.dst).getResultNumber()});
        if (portPair == portPairs.end()) {
          pairs.try_emplace(edge.dst, live(edge.dst), Value());
          break;
        }
        PortPair &pp = portPair->second;
        if (!pp.driven) {
          pp.driven = true;
          Block *body = getParentModule(src).getBodyBlock();
          drivePair(src.getLoc(), body->getArgument(pp.baseIdx),
                    body->getArgument(pp.enIdx), baseClk, enable);
        }
        auto *inst = liveInstance(edge.op);
        pairs.try_emplace(edge.dst, inst->getResult(pp.baseIdx),
                          inst->getResult(pp.enIdx));
        break;
      }
      }
    }
  }
}

void GatedClockConversion::rewriteRoot(const Root &root, Value baseClk,
                                       Value enable) {
  TypeSwitch<Operation *>(root.op)
      .Case<RefForceOp, RefReleaseOp>([&](auto op) {
        op.getClockMutable().assign(getDominatingValue(op, baseClk));
        ImplicitLocOpBuilder b(op.getLoc(), op);
        op.getPredicateMutable().assign(b.createOrFold<AndPrimOp>(
            op.getPredicate(), getDominatingValue(op, enable)));
      })
      .Case<RegOp, RegResetOp>([&](auto reg) {
        // Hold the register while the gate is closed.
        reg.getClockValMutable().assign(getDominatingValue(reg, baseClk));
        FConnectLike write = root.dataWrite;
        ImplicitLocOpBuilder b(write.getLoc(), write);
        Value newRhs = b.createOrFold<MuxPrimOp>(
            getDominatingValue(write, enable), write.getSrc(), reg.getData());
        write->setOperand(1, newRhs);
      })
      .Default([](Operation *) {
        llvm_unreachable("rejected by addRoot() or checkRoots()");
      });
}

void GatedClockConversion::rewriteRoots() {
  for (const Root &root : roots) {
    if (!root.isSinkable())
      continue;
    auto it = pairs.find(root.key);
    if (it == pairs.end())
      continue;
    auto [baseClk, enable] = it->second;
    if (enable)
      rewriteRoot(root, baseClk, enable);
  }
}

//===----------------------------------------------------------------------===//
// Clock alias analysis
//===----------------------------------------------------------------------===//

using ClockAliasAnalysis = GatedClockConversion::ClockAliasAnalysis;

unsigned ClockAliasAnalysis::find(unsigned node) const {
  unsigned root = node;
  while (nodes[root].parent != root)
    root = nodes[root].parent;
  // Path compression.
  while (nodes[node].parent != root)
    node = std::exchange(nodes[node].parent, root);
  return root;
}

unsigned ClockAliasAnalysis::track(Value v) {
  auto [it, inserted] = nodeOf.try_emplace(v, nodes.size());
  if (inserted) {
    nodes.push_back({v, it->second});
    members[it->second].push_back(it->second);
  }
  return it->second;
}

bool ClockAliasAnalysis::unionNodes(unsigned a, unsigned b) {
  a = find(a);
  b = find(b);
  if (a == b)
    return false;
  // Union by size.
  if (members[a].size() < members[b].size())
    std::swap(a, b);
  nodes[b].parent = a;
  auto small = members.find(b);
  members[a].append(small->second);
  members.erase(small);
  return true;
}

bool ClockAliasAnalysis::unionClocks(Value a, Value b) {
  if (!a || !b)
    return false;
  unsigned na = track(a);
  return unionNodes(na, track(b));
}

void ClockAliasAnalysis::clear() {
  nodes.clear();
  nodeOf.clear();
  members.clear();
  classBaseClock.clear();
}

Value ClockAliasAnalysis::getRepresentative(Value v) const {
  unsigned node = lookup(v);
  if (node == kNoNode)
    return Value();
  return nodes[members.find(find(node))->second.front()].value;
}

Value ClockAliasAnalysis::getBaseClock(Value v) const {
  unsigned node = lookup(v);
  return node == kNoNode ? Value() : classBaseClock.lookup(find(node));
}

SmallVector<Value> ClockAliasAnalysis::aliasSet(Value v) const {
  unsigned node = lookup(v);
  if (node == kNoNode)
    return {};
  return llvm::map_to_vector(members.find(find(node))->second,
                             [&](unsigned m) { return nodes[m].value; });
}

void ClockAliasAnalysis::setBaseClocks(ArrayRef<Value> baseClks) {
  for (Value base : baseClks)
    classBaseClock.try_emplace(find(track(base)), base);
}

void ClockAliasAnalysis::build(InstanceGraph &ig,
                               ArrayRef<std::pair<Value, Value>> shadowPairs) {
  clear();
  SmallVector<Value> baseClks;
  // A clock aliases its driver, and a gate aliases its input.
  auto visit = [&](Value v) {
    if (!type_isa<ClockType>(v.getType()))
      return;
    track(v);
    if (auto gate = v.getDefiningOp<ClockGateIntrinsicOp>())
      unionClocks(v, gate.getInput());
    else if (Value driver = clockDriver(v); driver && driver != v)
      unionClocks(v, driver);
    else if (driver && isBaseClock(v, ig))
      baseClks.push_back(v);
  };
  for (auto *node : ig) {
    auto mod = dyn_cast_or_null<FModuleOp>(node->getModule().getOperation());
    if (!mod)
      continue;
    Block *body = mod.getBodyBlock();
    for (BlockArgument arg : body->getArguments())
      visit(arg);
    body->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      for (Value result : op->getResults())
        visit(result);
      // Drivers cannot be traced through nested block arguments.
      if (llvm::any_of(op->getRegions(), [](Region &region) {
            return !region.empty() && region.front().getNumArguments();
          }))
        return WalkResult::skip();
      return WalkResult::advance();
    });
  }
  for (auto [a, b] : shadowPairs)
    unionClocks(a, b);
  closeOverInstances(ig);
  setBaseClocks(baseClks);
}

void ClockAliasAnalysis::closeOverInstances(InstanceGraph &ig) {
  // The rules are monotone, so a worklist reaches the least fixed point.
  std::deque<Value> worklist;
  DenseSet<Value> queued;
  auto enqueue = [&](Value v) {
    if (queued.insert(v).second)
      worklist.push_back(v);
  };
  for (const Node &node : nodes)
    enqueue(node.value);

  // A newly satisfied rule reads a value of each merged class, so requeueing
  // the smaller class suffices.
  auto merge = [&](Value a, Value b) {
    for (Value v : {a, b})
      if (!isTracked(v)) {
        track(v);
        enqueue(v);
      }
    unsigned ra = find(lookup(a)), rb = find(lookup(b));
    if (ra == rb)
      return;
    auto &ma = members[ra], &mb = members[rb];
    for (unsigned m : ma.size() <= mb.size() ? ma : mb)
      enqueue(nodes[m].value);
    unionNodes(ra, rb);
  };

  // Empty if some instance is not an `InstanceOp`.
  DenseMap<Operation *, SmallVector<InstanceOp>> instanceCache;
  auto instancesOf = [&](FModuleOp mod) -> MutableArrayRef<InstanceOp> {
    auto [it, inserted] = instanceCache.try_emplace(mod);
    if (!inserted)
      return it->second;
    for (auto *use : ig.lookup(mod)->uses()) {
      auto inst = use->getInstance<InstanceOp>();
      if (!inst) {
        it->second.clear();
        break;
      }
      it->second.push_back(inst);
    }
    return it->second;
  };

  // An input port joins its callers' class only if all instances agree; an
  // output port is related per instance.
  auto crossBoundary = [&](Value v) {
    FModuleOp mod;
    unsigned idx = 0;
    bool isArg = false;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      mod = dyn_cast_or_null<FModuleOp>(arg.getOwner()->getParentOp());
      idx = arg.getArgNumber();
      isArg = true;
    } else if (auto inst = v.getDefiningOp<InstanceOp>()) {
      mod = dyn_cast_or_null<FModuleOp>(
          inst.getReferencedModule(ig).getOperation());
      idx = cast<OpResult>(v).getResultNumber();
    }
    if (!mod)
      return;
    auto insts = instancesOf(mod);
    if (insts.empty())
      return;
    Block *body = mod.getBodyBlock();
    BlockArgument arg = body->getArgument(idx);
    if (!isTracked(arg))
      return;
    if (insts.size() == 1) {
      merge(arg, insts[0].getResult(idx));
      return;
    }
    bool isInput = mod.getPortDirection(idx) == Direction::In;
    if (isInput) {
      Value first = insts[0].getResult(idx);
      if (llvm::all_of(insts, [&](InstanceOp inst) {
            Value driver = inst.getResult(idx);
            return isTracked(driver) && alias(first, driver);
          }))
        merge(arg, first);
    }
    // Output-to-input relations only read the child's classes.
    if (!isArg)
      return;
    for (BlockArgument other : body->getArguments()) {
      unsigned otherIdx = other.getArgNumber();
      bool otherIsInput = mod.getPortDirection(otherIdx) == Direction::In;
      if (otherIsInput == isInput || !alias(arg, other))
        continue;
      unsigned outIdx = isInput ? otherIdx : idx;
      unsigned inIdx = isInput ? idx : otherIdx;
      for (auto inst : insts)
        merge(inst.getResult(outIdx), inst.getResult(inIdx));
    }
  };

  unsigned visits = 0;
  while (!worklist.empty()) {
    Value v = worklist.front();
    worklist.pop_front();
    queued.erase(v);
    ++visits;
    crossBoundary(v);
  }
  LLVM_DEBUG(llvm::dbgs() << "[closeOverInstances] " << visits << " visits, "
                          << members.size() << " classes\n");
}

static std::string getClockName(Value v) {
  auto [name, rootKnown] = getFieldName(getFieldRefFromValue(v));
  if (!rootKnown) {
    auto result = cast<OpResult>(v);
    name = ("<" + result.getOwner()->getName().getStringRef() + "#" +
            Twine(result.getResultNumber()) + ">")
               .str();
  }
  return (getParentModule(v).getModuleName() + "." + name).str();
}

void ClockAliasAnalysis::print(llvm::raw_ostream &os) const {
  SmallVector<std::string> lines;
  for (const auto &[root, classMembers] : members) {
    SmallVector<std::string> names;
    for (unsigned m : classMembers)
      names.push_back(getClockName(nodes[m].value));
    llvm::sort(names);
    Value base = classBaseClock.lookup(root);
    lines.push_back(
        "clock-alias-class: base=" + (base ? getClockName(base) : "<none>") +
        " members=[" + llvm::join(names, ", ") + "]");
  }
  llvm::sort(lines);
  for (auto &line : lines)
    os << line << "\n";
}

void GatedClockConversion::buildClockAliases() {
  // Input pairs are related in the module, output pairs at every instance.
  SmallVector<std::pair<Value, Value>> shadowPairs;
  for (auto &[key, portPair] : portPairs) {
    auto [mod, portIdx] = key;
    if (portPair.dir == Direction::In) {
      Block *body = mod.getBodyBlock();
      shadowPairs.emplace_back(body->getArgument(portPair.baseIdx),
                               body->getArgument(portIdx));
      continue;
    }
    for (auto *use : ig.lookup(mod)->uses())
      if (auto inst = use->getInstance<InstanceOp>())
        shadowPairs.emplace_back(inst.getResult(portPair.baseIdx),
                                 inst.getResult(portIdx));
  }
  aliases.build(ig, shadowPairs);
  aliasesBuilt = true;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

LogicalResult GatedClockConversion::run() {
  if (roots.empty())
    return success();
  context = roots[0].op->getContext();
  clockType = ClockType::get(context);
  u1Type = UIntType::get(context, 1);
  aliasesBuilt = false;

  if (failed(analyze()))
    return failure();
  LLVM_DEBUG(dump());
  markGated();
  if (failed(checkRoots()))
    return failure();

  SmallVector<Value> order = selectPortPairs();
  insertPorts();
  materialize(order);
  rewriteRoots();
  for (auto oldInst : deadInstances)
    oldInst.erase();
  buildClockAliases();

  roots.clear();
  analyzed.clear();
  srcToDstClocks.clear();
  baseClks.clear();
  gated.clear();
  portPairs.clear();
  pairs.clear();
  constU1Cache.clear();
  instClones.clear();
  deadInstances.clear();
  dominatingValues.clear();
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
      edge.op->print(llvm::dbgs());
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
