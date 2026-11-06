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
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

#define DEBUG_TYPE "firrtl-gated-clock-conversion"

using namespace circt;
using namespace firrtl;

static StringRef edgeKindName(EdgeKind kind) {
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

/// `ns` holds the module's port names, so the new names do not collide with
/// them or with each other.
static std::pair<PortInfo, PortInfo>
makeGatedClockPortInfos(MLIRContext *ctx, Namespace &ns, StringRef tag,
                        Direction dir, Location loc, Type clockType,
                        Type u1Type) {
  return {
      PortInfo(StringAttr::get(ctx, ns.newName("_gatedClock_baseClock_" + tag)),
               clockType, dir, /*symName=*/StringAttr(), loc),
      PortInfo(StringAttr::get(ctx, ns.newName("_gatedClock_enable_" + tag)),
               u1Type, dir, /*symName=*/StringAttr(), loc)};
}

static FModuleOp getParentModule(Value value) {
  if (isa<BlockArgument>(value))
    return cast<FModuleOp>(value.getParentBlock()->getParentOp());
  return value.getDefiningOp()->getParentOfType<FModuleOp>();
}

/// The `*_initial` force/release variants have no clock, so they are not
/// roots.
static Value clockOperandOf(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<RefForceOp, RefReleaseOp>([](auto op) { return op.getClock(); })
      .Case<RegOp, RegResetOp>([](auto op) { return op.getClockVal(); })
      .Case<ClockGateIntrinsicOp>([](auto op) { return op.getInput(); })
      .Default([](auto) { return Value(); });
}

/// The driver of `clk` through wires, nodes, casts and the input ports of
/// instances; null if undriven.
static Value clockDriver(Value clk) {
  SmallPtrSet<Value, 4> choicePorts;
  while (true) {
    // Look through casts; if that ends at a non-clock value, the clock is
    // created by a cast, so stop at the first cast instead.
    Value driver = getModuleScopedDriver(clk, /*lookThroughWires=*/true,
                                         /*lookThroughNodes=*/true,
                                         /*lookThroughCasts=*/true);
    // `asClock(asUInt(clk))` is `clk`, but `asClock(u)` is a clock of its own.
    if (driver && !type_isa<ClockType>(driver.getType()))
      driver = getModuleScopedDriver(clk, /*lookThroughWires=*/true,
                                     /*lookThroughNodes=*/true,
                                     /*lookThroughCasts=*/false);
    // `getModuleScopedDriver` looks through the input ports of an `InstanceOp`
    // only, so continue from the connect driving an `instance_choice` input.
    // A port reached twice is in a loop of wires, and is left as the driver.
    auto result = dyn_cast_or_null<OpResult>(driver);
    auto choice = result ? dyn_cast<InstanceChoiceOp>(result.getOwner())
                         : InstanceChoiceOp();
    if (!choice ||
        choice.getPortDirection(result.getResultNumber()) != Direction::In ||
        !choicePorts.insert(driver).second)
      return driver;
    clk = getDriverFromConnect(driver);
    if (!clk)
      return {};
  }
}

/// A driver-normal clock that is neither a gate nor passed in through a port.
static bool isBaseClock(Value clk, InstanceGraph &ig) {
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
                                       ClockGateIntrinsicOp gate) {
  if (!upstream)
    return gateEn;
  // No single point may see both enables, e.g. when `gate` is in a layer block
  // and `upstream` is defined after it in the parent block, so carry
  // `upstream` to the gate instead.
  ImplicitLocOpBuilder builder(gate.getLoc(), gate);
  return builder.createOrFold<AndPrimOp>(bounceValue(gate, upstream), gateEn);
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
  if (!en)
    en = getOrCreateConstU1One(getParentModule(dstClk));
  // At the end of the block, after any value defined in it. A value defined
  // later in an enclosing block, e.g. when the instance is in a layer block,
  // is carried in by a bounce wire.
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, dstClk.getParentBlock());
  for (auto [dst, src] : {std::pair(dstClk, clk), std::pair(dstEn, en)}) {
    auto connect = MatchingConnectOp::create(builder, dst, src);
    connect.getSrcMutable().assign(bounceValue(connect, src));
  }
}

Value GatedClockConversion::bounceValue(Operation *user, Value v) {
  if (domInfo.properlyDominates(v, user))
    return v;

  // `user` reads `v` through a wire driven later in the block, so carry `v` to
  // it through a bounce wire declared at the top of the block.
  Block *defBlock = v.getParentBlock();
  assert(defBlock->findAncestorOpInBlock(*user) &&
         "a value is used within the block that defines it");
  auto [it, inserted] = bounceWires.try_emplace(v);
  if (inserted) {
    ImplicitLocOpBuilder builder(v.getLoc(), context);
    builder.setInsertionPointToStart(defBlock);
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
    // The enable is sunk as a mux over the whole register, which a field-wise
    // write or a non-passive aggregate would not support.
    if (!type_cast<FIRRTLType>(regData.getType()).isGround())
      return op->emitError("gated clock conversion: expected a ground-typed "
                           "register (run after firrtl-lower-types); found ")
             << regData.getType();
    // Whether the enable must gate the reset depends on whether it is sync.
    if (auto regReset = dyn_cast<RegResetOp>(op);
        regReset && type_isa<ResetType>(regReset.getResetSignal().getType()))
      return op->emitError("gated clock conversion: register has an uninferred "
                           "reset type (run after firrtl-infer-resets)");
    unsigned numWrites = 0;
    for (auto &use : regData.getUses()) {
      // Only count the destination operand, so a self-connect `connect r, r`
      // (which uses the register data twice) is one write, not two.
      auto fconn = dyn_cast<FConnectLike>(use.getOwner());
      if (fconn && use.get() == fconn.getDest() &&
          use.getOperandNumber() == 0) {
        ++numWrites;
        root.dataWrite = fconn;
      }
    }
    if (numWrites != 1)
      return op->emitError("gated clock conversion: expected exactly one "
                           "connect driving this register (run after "
                           "firrtl-expand-whens); found ")
             << numWrites;
  }
  roots.push_back(root);
  return success();
}

LogicalResult GatedClockConversion::analyze() {
  // Construct the clock driver graph, by tracing back from the root clocks, to
  // find the clock gate ops on the path to the base clock. Any clock whose
  // driver cannot be analyzed/traced back is a base clock. For example, block
  // argument of top module, result of an extern module, result of a mux op are
  // all base clocks.
  LogicalResult result = success();
  SmallVector<Value> worklist;

  DenseSet<Value> addedToWorklist;
  // Without a driver there is no pair to give the clock, and a caller of an
  // input port pair would be left not driving it.
  auto reportUndriven = [&](Value clk) {
    mlir::emitError(clk.getLoc())
        << "gated clock conversion: this clock is not driven; run this "
           "utility after firrtl-expand-whens and firrtl-check-init";
    result = failure();
  };

  auto enqueue = [&](Value clk) {
    if (addedToWorklist.insert(clk).second)
      worklist.push_back(clk);
  };

  auto addEdge = [&](Value dstClk, Value srcClk, Operation *op, EdgeKind kind) {
    if (Value driver = clockDriver(srcClk)) {
      LLVM_DEBUG(llvm::dbgs() << "  edge kind=" << edgeKindName(kind) << "\n");
      srcToDstClocks[driver].push_back({dstClk, op, kind});
      enqueue(driver);
    } else
      reportUndriven(srcClk);
  };

  auto visit = [&](Value clk) {
    if (auto blockArg = dyn_cast<BlockArgument>(clk)) {
      auto mod = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
      unsigned portIdx = blockArg.getArgNumber();
      auto *node = ig.lookup(mod);
      // A public module's interface cannot change, since it may also be
      // instantiated outside the circuit, so its ports are base clocks.
      if (node->noUses() || mod.isPublic()) {
        baseClks.push_back(clk);
      } else {
        // Every caller is traced, including an `instance_choice`, so whether
        // the port is gated is known. `checkSelection()` rejects a port pair
        // on a module that is not only used by `InstanceOp`s. An
        // `instance_choice` is a use once per target, so trace it once.
        SmallPtrSet<Operation *, 4> callers;
        for (auto *use : node->uses()) {
          auto caller = cast<FInstanceLike>(use->getInstance().getOperation());
          if (callers.insert(caller).second)
            addEdge(clk, caller->getResult(portIdx), caller,
                    EdgeKind::InstanceIn);
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
      if (childMod && !childMod.isPublic()) {
        unsigned portIdx = cast<OpResult>(clk).getResultNumber();
        addEdge(clk, childMod.getBodyBlock()->getArgument(portIdx), inst,
                EdgeKind::InstanceOut);
      } else
        baseClks.push_back(clk);
      return;
    }

    // Any other op that produces a clock, is considered a base clock, this
    // could even be a mux.
    baseClks.push_back(clk);
  };

  for (Root &root : roots) {
    if ((root.key = clockDriver(root.clock)))
      enqueue(root.key);
    else
      reportUndriven(root.clock);
  }

  while (!worklist.empty())
    visit(worklist.pop_back_val());
  return result;
}

SmallVector<Value> GatedClockConversion::markGatedAndSelectPortPairs() {
  // Forward walk from the base clocks to every clock derived from them,
  // recording in `gated` whether each is in a gated clock domain, and choosing
  // the port pairs that carry a gated clock across a module boundary.
  //
  // The graph can have cycles and multiple incoming edges, since a module's
  // values are shared by all of its instances, so a clock's bit is not final
  // when it is first visited. A clock is requeued when its bit flips, so each
  // is visited at most twice. Bits only flip from false to true and port pairs
  // are only added, so the last visit of each clock, which sees its final bit,
  // selects its port pairs.
  //
  // A requeued clock discovers nothing new, so `order` is the clocks in
  // first-discovery order. Each clock is discovered by a predecessor that is
  // already in `order`, which `materialize()` relies on.
  SmallVector<Value> order, worklist;
  auto discover = [&](Value clk, bool isGated) {
    auto [it, inserted] = gated.try_emplace(clk, isGated);
    if (inserted) {
      order.push_back(clk);
      worklist.push_back(clk);
    } else if (isGated && !it->second) {
      it->second = true;
      worklist.push_back(clk);
    }
  };
  for (Value baseClk : baseClks)
    discover(baseClk, false);

  while (!worklist.empty()) {
    Value src = worklist.pop_back_val();
    auto edges = srcToDstClocks.find(src);
    if (edges == srcToDstClocks.end())
      continue;
    bool srcGated = gated.lookup(src);
    for (const ClockEdge &edge : edges->second) {
      bool dstGated = srcGated || edge.kind == EdgeKind::Gate;
      if (edge.kind == EdgeKind::InstanceIn && dstGated) {
        auto arg = cast<BlockArgument>(edge.dst);
        portPairs.try_emplace({cast<FModuleOp>(arg.getOwner()->getParentOp()),
                               arg.getArgNumber()},
                              PortPair{Direction::In});
      } else if (edge.kind == EdgeKind::InstanceOut && srcGated) {
        portPairs.try_emplace(
            {getParentModule(src), cast<OpResult>(edge.dst).getResultNumber()},
            PortPair{Direction::Out});
      }
      discover(edge.dst, dstGated);
    }
  }
  return order;
}

LogicalResult GatedClockConversion::checkSelection() {
  // A clock missing from `gated` is not reachable from any base clock, so it is
  // driven by a clock loop and has no pair. Nothing would rebase a root on such
  // a clock, or drive a caller's half of an input port pair from it.
  LogicalResult result = success();
  DenseSet<Value> reportedClocks;
  for (const Root &root : roots) {
    if (gated.contains(root.key) || !reportedClocks.insert(root.key).second)
      continue;
    mlir::emitError(root.key.getLoc())
        << "gated clock conversion: this clock is driven by a clock loop, and "
           "is not reachable from any base clock";
    result = failure();
  }

  // Ports can only be added to the instances of a module if they are all
  // `InstanceOp`s. This is the only place that checks it, so a module used by
  // an `instance_choice` is only rejected if it needs a port pair, and each
  // such use is reported once.
  DenseSet<Operation *> reportedInsts;
  for (auto &[key, portPair] : portPairs) {
    auto [mod, portIdx] = key;
    for (auto *use : ig.lookup(mod)->uses()) {
      Operation *instOp = use->getInstance().getOperation();
      auto inst = dyn_cast<InstanceOp>(instOp);
      if (!inst) {
        if (reportedInsts.insert(instOp).second) {
          instOp->emitError("can only handle InstanceOp");
          result = failure();
        }
        continue;
      }
      if (portPair.dir != Direction::In ||
          gated.contains(clockDriver(inst.getResult(portIdx))))
        continue;
      inst.emitError("gated clock conversion: clock port ")
          << mod.getPortNameAttr(portIdx)
          << " of this instance is driven by a clock loop, and is not "
             "reachable from any base clock";
      result = failure();
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Mutation
//===----------------------------------------------------------------------===//

void GatedClockConversion::insertPorts() {
  llvm::MapVector<FModuleOp, SmallVector<std::pair<unsigned, PortPair *>>>
      perModule;
  for (auto &[key, portPair] : portPairs)
    perModule[key.first].push_back({key.second, &portPair});

  for (auto &[mod, modPairs] : perModule) {
    // One call per module, so each instance is re-created once.
    llvm::sort(modPairs, llvm::less_first());
    const unsigned origNumPorts = mod.getNumPorts();
    Namespace portNames;
    for (auto name : mod.getPortNames())
      portNames.add(cast<StringAttr>(name).getValue());
    SmallVector<std::pair<unsigned, PortInfo>> newPorts;
    for (auto [portIdx, portPair] : modPairs) {
      portPair->baseIdx = origNumPorts + newPorts.size();
      portPair->enIdx = portPair->baseIdx + 1;
      auto [baseInfo, enableInfo] = makeGatedClockPortInfos(
          context, portNames, mod.getPortName(portIdx), portPair->dir,
          mod.getPortLocation(portIdx), clockType, u1Type);
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
    pairs[baseClk] = {live(baseClk), Value()};

  for (Value src : order) {
    auto [baseClk, enable] = pairs.at(src);
    assert(!enable == !gated.lookup(src) && "a clock has an enable iff gated");
    auto edges = srcToDstClocks.find(src);
    if (edges == srcToDstClocks.end())
      continue;
    for (const ClockEdge &edge : edges->second) {
      switch (edge.kind) {
      case EdgeKind::Gate: {
        auto gate = edge.gate();
        [[maybe_unused]] bool inserted =
            pairs
                .try_emplace(edge.dst, baseClk,
                             andEnables(enable, gateEnableOf(gate), gate))
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
        op.getClockMutable().assign(bounceValue(op, baseClk));
        ImplicitLocOpBuilder b(op.getLoc(), op);
        op.getPredicateMutable().assign(b.createOrFold<AndPrimOp>(
            op.getPredicate(), bounceValue(op, enable)));
      })
      .Case<ClockGateIntrinsicOp>([&](auto gate) {
        gate.getInputMutable().assign(bounceValue(gate, baseClk));
        gate.getEnableMutable().assign(
            andEnables(enable, gate.getEnable(), gate));
        if (Value testEnable = gate.getTestEnable())
          gate.getTestEnableMutable().assign(
              andEnables(enable, testEnable, gate));
      })
      .Case<RegOp, RegResetOp>([&](auto reg) {
        // Hold the register while the gate is closed.
        reg.getClockValMutable().assign(bounceValue(reg, baseClk));
        FConnectLike write = root.dataWrite;
        ImplicitLocOpBuilder b(write.getLoc(), write);
        Value newRhs = b.createOrFold<MuxPrimOp>(bounceValue(write, enable),
                                                 write.getSrc(), reg.getData());
        write->setOperand(1, newRhs);

        // A synchronous reset only takes effect on a gated clock edge, so it
        // must be held off while the gate is closed too. An asynchronous reset
        // does not depend on the clock and is left alone. `addRoot()` rejected
        // uninferred resets.
        if (auto regReset = dyn_cast<RegResetOp>(reg.getOperation())) {
          Value resetSignal = regReset.getResetSignal();
          assert(!type_isa<ResetType>(resetSignal.getType()) &&
                 "uninferred reset");
          if (type_isa<UIntType>(resetSignal.getType())) {
            ImplicitLocOpBuilder rb(regReset.getLoc(), regReset);
            regReset.getResetSignalMutable().assign(rb.createOrFold<AndPrimOp>(
                resetSignal, bounceValue(regReset, enable)));
          }
        }
      })
      .Default(
          [](Operation *) { llvm_unreachable("unsupported root operation"); });
}

void GatedClockConversion::rewriteRoots() {
  for (const Root &root : roots) {
    // `checkSelection()` ensured every root clock is reachable.
    auto [baseClk, enable] = pairs.at(root.key);
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

Value ClockAliasAnalysis::atInstance(Value v, InstanceOp inst) const {
  // Aliases within a module hold in every instance of it.
  for (BlockArgument arg : getParentModule(v).getArguments())
    if (alias(v, arg))
      return inst.getResult(arg.getArgNumber());
  return v;
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

void GatedClockConversion::clear() {
  roots.clear();
  srcToDstClocks.clear();
  baseClks.clear();
  gated.clear();
  portPairs.clear();
  pairs.clear();
  constU1Cache.clear();
  instClones.clear();
  deadInstances.clear();
  bounceWires.clear();
  domInfo.invalidate();
}

LogicalResult GatedClockConversion::run() {
  // Every exit, including a failed analysis, leaves no state behind.
  llvm::scope_exit clearOnExit([&] { clear(); });
  if (roots.empty())
    return success();
  context = roots[0].op->getContext();
  clockType = ClockType::get(context);
  u1Type = UIntType::get(context, 1);
  aliasesBuilt = false;

  if (failed(analyze()))
    return failure();
  LLVM_DEBUG(dump());
  SmallVector<Value> order = markGatedAndSelectPortPairs();
  if (failed(checkSelection()))
    return failure();
  insertPorts();
  materialize(order);
  rewriteRoots();
  for (auto oldInst : deadInstances)
    oldInst.erase();
  buildClockAliases();
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
