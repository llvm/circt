//===- ProbesToSignals.cpp - Probes to Signals ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ProbesToSignals pass.  This pass replaces probes with
// signals of the same type.  This is not considered a lowering but a
// behavior-changing transformation that may break ABI compatibility anywhere
// probes are used relevant to ABI.
//
// Read-only probes lower to plain wire connections.  Force/release on RWProbes
// is synthesized into a per-probe state machine: a `forced` flag register and
// a `forcedValue` register encode the effect of the priority-ordered accesses,
// and an override mux is injected at the target's connect point.
//
// Gated-clock force/release: when the access's clock or the target register's
// clock comes from a `firrtl.int.clock_gate`, the gate enable is folded into
// the synchronous predicate so the synthesized state runs on the free-running
// base clock.  Cross-module gates are handled by GatedClockConversion, which
// traces each access's clock across module boundaries and plumbs paired
// `_pts_baseClock_*` / `_pts_gateEnable_*` ports along the path so the access
// can reference them as local SSA values.
//
// Pre-requisites for complete conversion:
// * LowerOpenAggs
//   - Simplifies this pass, Probes are always separate.
// * ExpandWhens
//   - ref.define is "static single connect", and FIRRTL does not have
//     an equivalent for hardware connections.  As a result, probes sent out
//     from under a "when" cannot be represented currently.
//
// Suggested:
// * Inference passes, especially width inference.  Probes infer slightly
//   differently than non-probes do (must have same width along the chain).
//
// Colored probes are not supported.
// Specialize layers on or off to remove colored probes first.
//
// Debug ports on FIRRTL memories are not currently supported,
// but CHIRRTL debug ports are handled.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/GatedClockConversion.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "firrtl-probes-to-signals"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_PROBESTOSIGNALS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Probes to Signals
//===----------------------------------------------------------------------===//

namespace {

/// Return the parent FModuleOp of a given value: for a BlockArgument the module
/// owning its block, otherwise the module containing its defining op.
static FModuleOp getParentModule(Value value) {
  if (isa<BlockArgument>(value))
    return cast<FModuleOp>(value.getParentBlock()->getParentOp());
  return value.getDefiningOp()->getParentOfType<FModuleOp>();
}

/// One force or release access targeting a single RWProbe.  `forceValue`
/// distinguishes the two: `nullopt` means release.  `clock` is null for
/// `force_initial` / `release_initial` — those have no synchronous component
/// and only ride along on a sibling clocked access's state machine.
struct ForceReleaseAccess {
  Operation *op;
  Value predicate;
  std::optional<Value> forceValue;
  Value clock;

  bool isForce() const { return forceValue.has_value(); }
};

class ProbeVisitor : public FIRRTLVisitor<ProbeVisitor, LogicalResult> {
public:
  ProbeVisitor(hw::InnerRefNamespace &irn) : irn(irn) {}

  /// Entrypoint.
  LogicalResult visit(FModuleLike mod);

  /// Reduced local force/release control for a probe.  `clk` is null when every
  /// contributing access is an `_initial` variant.
  struct LocalCtrl {
    Value forceActive;
    Value releaseActive;
    Value forcedValue;
    Value clk;
  };

  /// A forceable RWProbe exported through a module port.  The port insertion
  /// and inbound driving are deferred to the sequential post-pass because a
  /// module's port count and its instances' port counts must be mutated
  /// together to stay in lockstep, which is unsafe under the parallel walk.
  struct ExportEntry {
    unsigned portIdx;          ///< exported port index (== instance result idx)
    StringAttr portName;       ///< original port name (-> "<name>_force_ctrl")
    Location portLoc;          ///< location for the new port
    FIRRTLBaseType probedType; ///< probed (hardware) type
    WireOp controlWire;        ///< SM control-bundle wire data in this module
    LocalCtrl local;           ///< reduced local control; all-null iff no local
                               ///< force (treated as constant 0 in the merge)
  };

  /// A forceable RWProbe consumed *from* an instance in this (parent) module.
  /// The instance's freshly-inserted `<name>_force_ctrl` input port is
  /// connected to the parent's forwarding control wire by the sequential
  /// post-pass.
  struct InstancePlumb {
    Operation *inst;      ///< type-converted instance (live into the post-pass)
    unsigned resultIdx;   ///< forceable probe result index on the instance
    Value forwardingWire; ///< control-bundle wire in the parent driving it
  };

  /// How a forceable target's synthesized state machine gets its control.
  enum class ForceCategory {
    /// Purely local: registers driven directly from reduced SSA (no wire).
    Local,
    /// Exported through a module port: control merged with an inbound
    /// `_force_ctrl` bundle by the sequential post-pass.
    Exported,
    /// Consumed from a child instance: the real SM lives in the child; drive
    /// the parent's forwarding control-bundle wire from reduced local control.
    InstanceForwarded,
  };

  /// Per-target force/release plan: records *what* to materialize (category +
  /// the reduced control + the target metadata) but emits no hardware itself.
  struct ForcePlan {
    ForceCategory category;
    Value target;              ///< canonical hardware value being forced
    FIRRTLBaseType probedType; ///< probed (hardware) type of `target`
    /// Reduced local control (all-null iff no local force, e.g. a probe forced
    /// only from outside this module).
    LocalCtrl local;
    /// Instance-forwarded only: the parent's forwarding control-bundle wire.
    WireOp forwardingWire;
  };

  /// Exported forceable probes / instance force-control wires discovered while
  /// visiting this module, read by the pass after `visit` returns and replayed
  /// in the sequential post-pass.
  SmallVector<ExportEntry> exportWork;
  SmallVector<InstancePlumb> instancePlumb;

  using FIRRTLVisitor<ProbeVisitor, LogicalResult>::visitDecl;
  using FIRRTLVisitor<ProbeVisitor, LogicalResult>::visitExpr;
  using FIRRTLVisitor<ProbeVisitor, LogicalResult>::visitStmt;

  //===--------------------------------------------------------------------===//
  // Type conversion
  //===--------------------------------------------------------------------===//

  /// Return the converted type, null if same, failure on error.
  static FailureOr<Type> convertType(Type type, Location loc) {
    auto err = [type, loc](const Twine &message) {
      return mlir::emitError(loc, message) << ", cannot convert type " << type;
    };
    if (isa<OpenBundleType, OpenVectorType>(type))
      return err("open aggregates not supported");

    auto refType = dyn_cast<RefType>(type);
    if (!refType)
      return Type();

    if (refType.getLayer())
      return err("layer-colored probes not supported");

    // Otherwise, this maps to the probed type.
    return refType.getType();
  }

  /// Return "target" type, or failure on error.
  static FailureOr<Type> mapType(Type type, Location loc) {
    auto newType = convertType(type, loc);
    if (failed(newType))
      return failure();
    return *newType ? *newType : type;
  }

  /// Map a range of types, return if changes needed.
  template <typename R>
  static FailureOr<bool> mapRange(R &&range, Location loc,
                                  SmallVectorImpl<Type> &newTypes) {
    newTypes.reserve(llvm::size(range));

    bool anyConverted = false;
    for (auto type : range) {
      auto conv = mapType(type, loc);
      if (failed(conv))
        return failure();
      newTypes.emplace_back(*conv);
      anyConverted |= *conv != type;
    }
    return anyConverted;
  }

  // CHIRRTL
  LogicalResult visitMemoryDebugPortOp(chirrtl::MemoryDebugPortOp op);

  // Visitors

  LogicalResult visitInvalidOp(Operation *op) {
    if (auto dbgPortOp = dyn_cast<chirrtl::MemoryDebugPortOp>(op))
      return visitMemoryDebugPortOp(dbgPortOp);

    return visitUnhandledOp(op);
  }
  LogicalResult visitUnhandledOp(Operation *op);

  /// Check declarations specifically before forwarding to unhandled.
  LogicalResult visitUnhandledDecl(Operation *op) {
    // Check for and handle active forceable declarations.
    if (auto fop = dyn_cast<Forceable>(op); fop && fop.isForceable())
      return visitActiveForceableDecl(fop);
    return visitUnhandledOp(op);
  }

  // Declarations

  LogicalResult visitDecl(MemOp op);
  LogicalResult visitDecl(WireOp op);
  LogicalResult visitActiveForceableDecl(Forceable fop);

  LogicalResult visitInstanceLike(FInstanceLike oldInst);
  LogicalResult visitDecl(InstanceOp op) { return visitInstanceLike(op); }
  LogicalResult visitDecl(InstanceChoiceOp op) { return visitInstanceLike(op); }

  // Probe operations.

  LogicalResult visitExpr(RWProbeOp op);
  LogicalResult visitExpr(RefCastOp op);
  LogicalResult visitExpr(RefResolveOp op);
  LogicalResult visitExpr(RefSendOp op);
  LogicalResult visitExpr(RefSubOp op);

  LogicalResult visitStmt(RefDefineOp op);

  // Force and release operations: collect for later synthesis.
  LogicalResult visitStmt(RefForceOp op);
  LogicalResult visitStmt(RefForceInitialOp op);
  LogicalResult visitStmt(RefReleaseOp op);
  LogicalResult visitStmt(RefReleaseInitialOp op);

private:
  /// Map from probe-typed Value's to their non-probe equivalent.
  DenseMap<Value, Value> probeToHWMap;
  DenseMap<Value, WireOp> targetToForceCtrlWire;

  /// Forceable operations to demote.
  SmallVector<Forceable> forceables;

  /// Operations to delete.
  SmallVector<Operation *> toDelete;

  /// Read-only copy of inner-ref namespace for resolving inner refs.
  hw::InnerRefNamespace &irn;

  /// Map from RWProbe values to all force/release accesses targeting them.
  MapVector<Value, SmallVector<ForceReleaseAccess>> forceReleaseMap;

  /// Reduced local control per probe (see `LocalCtrl`).
  DenseMap<Value, LocalCtrl> localCtrlMap;

  /// Per-target force/release plans produced by `analyzeForcePlans`, ordered
  /// deterministically (`forceReleaseMap` order).
  SmallVector<ForcePlan> forcePlans;

  /// Probes that are exported through a forceable port.  Their control wires
  /// are driven by the sequential post-pass (merging local + inbound), so
  /// `materializeForcePlans` must skip them to avoid a second driver.
  SmallPtrSet<Value, 8> exportedProbes;

  /// Cached types for creating force control bundles
  UIntType u1Type;
  ClockType clkType;

  /// Cache for bundle types to avoid creating duplicates
  DenseMap<Type, BundleType> bundleTypeCache;

  /// Reduce the priority-ordered force/release accesses for one target into a
  /// single `LocalCtrl`.  Builds the `forceWins` arbitration and the
  /// OR-reductions of the force/release predicates.  `builder`'s insertion
  /// point is used for the constants and the per-access
  /// `forceWins`/`forcedValue` muxes (inserted after each access op); the final
  /// AND/OR reductions are emitted at `builder`'s current point on return.
  /// `clk` is null when every contributing access is an `_initial` variant.
  LocalCtrl reduceAccesses(ImplicitLocOpBuilder &builder,
                           FIRRTLBaseType probedType,
                           ArrayRef<ForceReleaseAccess> accesses);

  /// Reduce every target's priority-ordered accesses (`forceReleaseMap`) into a
  /// `LocalCtrl` stored in `localCtrlMap`.  Emits the arbitration
  /// muxes/OR-reductions but makes no materialization decisions (registers,
  /// wires, ports).
  LogicalResult reduceForceReleaseControl(FModuleLike mod);

  /// Classify every forced target into a `ForcePlan` (`forcePlans`) recording
  /// its `ForceCategory` and reduced control, without emitting any hardware.
  /// Runs after the export loop has populated `exportedProbes` /
  /// `targetToForceCtrlWire`, so each target's category is known.
  void analyzeForcePlans();

  /// Realize each `ForcePlan`:
  ///   * Local             -> `buildStateMachineRegisters` (no control wire);
  ///   * InstanceForwarded -> drive the parent's forwarding bundle wire;
  ///   * Exported          -> skipped here (the sequential post-pass merges
  ///                          local + inbound so each field keeps one driver).
  LogicalResult materializeForcePlans();

  /// Synthesize the force/release state machine for a single forceable
  /// declaration. Creates the control bundle wire, registers for forced flag
  /// and forced value, and inserts the appropriate mux logic.  Returns the
  /// force control wire, or a null WireOp (with a diagnostic emitted) on error.
  WireOp synthesizeForceableStateMachine(FIRRTLBaseType probedType, Value data);

  /// Reduced force/release control fed into the state-machine registers.  `clk`
  /// is null when every contributing access is an `_initial` variant (which is
  /// diagnosed).
  struct StateMachineInputs {
    Value forceActive;
    Value releaseActive;
    Value forcedValue;
    Value clk;
  };

  /// Emit the two state-machine registers (`forced`, `forcedValue`) and the
  /// override mux at `data`'s connect point, driving the registers directly
  /// from the supplied control SSA values.  Used for purely-local forceable
  /// targets, which need no control-bundle wire at all.  Returns failure (with
  /// a diagnostic emitted) if `data` is read-only or `clk` is null.
  LogicalResult buildStateMachineRegisters(FIRRTLBaseType probedType,
                                           Value data,
                                           const StateMachineInputs &in);

  /// Return the control-bundle wire for a forceable target, synthesizing the
  /// state machine on first use.  Creation can be triggered by a local force,
  /// an exported forceable port, or an instance-result forwarding wire, so this
  /// single accessor keeps the "did we create it yet?" logic in one place.
  /// Returns a null WireOp (with a diagnostic already emitted) on error.
  WireOp getOrCreateStateMachine(FIRRTLBaseType probedType, Value hwVal) {
    auto it = targetToForceCtrlWire.find(hwVal);
    if (it != targetToForceCtrlWire.end())
      return it->second;
    auto wire = synthesizeForceableStateMachine(probedType, hwVal);
    if (!wire)
      return {};
    targetToForceCtrlWire[hwVal] = wire;
    return wire;
  }

  /// Create the force control bundle type with {forceActive, releaseActive,
  /// forcedValue, clk} fields. Uses cached u1Type and clkType, and caches
  /// the resulting bundle type to avoid creating duplicates.
  BundleType createForceCtrlBundleType(FIRRTLBaseType probedType) {
    auto it = bundleTypeCache.find(probedType);
    if (it != bundleTypeCache.end())
      return it->second;

    auto *ctx = u1Type.getContext();
    SmallVector<BundleType::BundleElement> elements = {
        {StringAttr::get(ctx, "forceActive"), /*isFlip=*/false, u1Type},
        {StringAttr::get(ctx, "releaseActive"), /*isFlip=*/false, u1Type},
        {StringAttr::get(ctx, "forcedValue"), /*isFlip=*/false, probedType},
        {StringAttr::get(ctx, "clk"), /*isFlip=*/false, clkType},
    };
    auto bundleType = BundleType::get(ctx, elements);

    bundleTypeCache[probedType] = bundleType;
    return bundleType;
  }
};

} // end namespace

//===----------------------------------------------------------------------===//
// Visitor: FModuleLike
//===----------------------------------------------------------------------===//

static Block *getBodyBlock(FModuleLike mod) {
  // Safety check for below, presently all modules have a region.
  assert(mod->getNumRegions() == 1);
  auto &blocks = mod->getRegion(0).getBlocks();
  return !blocks.empty() ? &blocks.front() : nullptr;
}

/// Visit a module, converting its ports and internals to use hardware signals
/// instead of probes.
LogicalResult ProbeVisitor::visit(FModuleLike mod) {
  auto *ctx = mod->getContext();
  u1Type = UIntType::get(ctx, 1);
  clkType = ClockType::get(ctx);
  // Ports -> new ports without probe-ness.
  // For all probe ports, insert non-probe duplex values to use
  // as their replacement while rewriting.  Only if has body.
  SmallVector<std::pair<size_t, WireOp>> wires;

  auto portTypes = mod.getPortTypes();
  auto portLocs = mod.getPortLocationsAttr().getAsRange<Location>();
  auto portNames = mod.getPortNamesAttr();
  SmallVector<Attribute> newPortTypes;

  // Collect RWProbe ports: for each port whose original type is an RWProbe,
  // record the port index so we can add a control bundle port.
  SmallVector<unsigned> rwProbePorts;

  wires.reserve(portTypes.size());
  newPortTypes.reserve(portTypes.size());
  auto *block = getBodyBlock(mod);
  bool portsToChange = false;
  for (auto [idx, typeAttr, loc] : llvm::enumerate(portTypes, portLocs)) {
    auto type = cast<TypeAttr>(typeAttr);
    auto conv = convertType(type.getValue(), loc);
    if (failed(conv))
      return failure();
    auto newType = *conv;

    if (newType) {
      portsToChange = true;
      newPortTypes.push_back(TypeAttr::get(newType));
      if (block) {
        auto builder = OpBuilder::atBlockBegin(block);
        wires.emplace_back(idx, WireOp::create(builder, loc, newType));
        probeToHWMap[block->getArgument(idx)] = wires.back().second.getData();
      }
      // If the original port was an RWProbe, record it so we can add an input
      // bundle port (<name>_force_ctrl) carrying {forceActive, releaseActive,
      // forcedValue} for driving the remote wire/register target.
      if (auto refType = dyn_cast<RefType>(type.getValue());
          refType.getForceable()) {
        rwProbePorts.push_back(idx);
      }

    } else
      newPortTypes.push_back(type);
  }

  // Update body, if present.
  if (block &&
      block
          ->walk<mlir::WalkOrder::PreOrder>(
              [&](Operation *op) -> WalkResult { return dispatchVisitor(op); })
          .wasInterrupted())
    return failure();

  // Reduce each target's accesses to a LocalCtrl before modifying signatures.
  if (failed(reduceForceReleaseControl(mod)))
    return failure();

  // Update signature and argument types.
  if (portsToChange) {
    mod.setPortTypesAttr(ArrayAttr::get(mod->getContext(), newPortTypes));

    if (block) {
      // We may also need to update the types on the block arguments.
      for (auto [arg, typeAttr] :
           llvm::zip_equal(block->getArguments(), newPortTypes))
        arg.setType(cast<TypeAttr>(typeAttr).getValue());

      // Drop the port stand-ins and RAUW to the block arguments.
      for (auto [idx, wire] : wires) {
        auto arg = block->getArgument(idx);
        wire.getData().replaceAllUsesWith(arg);
        wire.erase();
      }
    }
  }

  // For each forceable RWProbe port exported through a `ref.define`, gather the
  // work needed to plumb an inbound `<name>_force_ctrl` control port.  The
  // state machine is synthesized here (module-local, so parallel-safe), but the
  // port insertion and inbound driving are deferred to a sequential post-pass
  // (`runOnOperation`): a module's port count and its instances' port counts
  // must be mutated together to stay in lockstep, which is unsafe under the
  // parallel module walk.  See `ExportEntry`.
  if (block && !rwProbePorts.empty()) {
    for (unsigned portIdx : rwProbePorts) {
      auto rwProbe = block->getArgument(portIdx);
      // Find the ref.define that exports the local target out of this port.
      RefDefineOp refDef;
      for (auto *o : rwProbe.getUsers())
        if (auto rd = dyn_cast<RefDefineOp>(o)) {
          refDef = rd;
          break;
        }

      if (!refDef)
        // The port is consumed in a way we cannot route force control through
        // (e.g. only via ref.sub).  Diagnose rather than assert.
        return mod->emitError(
                   "forceable probe port cannot be lowered: no ref.define "
                   "exporting a local target for port ")
               << cast<StringAttr>(portNames[portIdx]).getValue();

      auto outSrc = refDef.getSrc();
      auto probedType = type_cast<FIRRTLBaseType>(
          cast<RefType>(cast<TypeAttr>(portTypes[portIdx]).getValue())
              .getType());

      // localCtrlMap is keyed by hardware values; translate the probe-typed
      // outSrc through probeToHWMap to get the canonical key.
      auto hwSrcIt = probeToHWMap.find(outSrc);
      assert(hwSrcIt != probeToHWMap.end() &&
             "exported forceable target has no hardware value");
      Value hwSrc = hwSrcIt->second;

      // Ensure the exported target has a state machine: a probe that is only
      // forced from outside (never locally) has no entry yet.
      if (!getOrCreateStateMachine(probedType, hwSrc))
        return failure();

      // Record the deferred port/instance plumbing for the sequential pass,
      // which is this probe's single driver (merging local + inbound).  The
      // local control is optional — when absent, an all-null LocalCtrl is
      // stored and treated as constant 0 in the uniform merge.
      exportedProbes.insert(hwSrc);
      auto localIt = localCtrlMap.find(hwSrc);
      exportWork.push_back(
          {portIdx, cast<StringAttr>(portNames[portIdx]),
           mod.getPortLocation(portIdx), probedType,
           targetToForceCtrlWire[hwSrc],
           localIt != localCtrlMap.end() ? localIt->second : LocalCtrl{}});
    }
  }

  // Now that the export loop has classified which targets are exported /
  // instance-forwarded, build a per-target `ForcePlan`.
  analyzeForcePlans();

  // Realize each plan (local SM registers, or drive an instance forwarding
  // wire; exported plans are handled by the sequential post-pass).
  if (failed(materializeForcePlans()))
    return failure();

  for (auto *op : llvm::reverse(toDelete))
    op->erase();

  // Demote forceable declarations: their force/release effect is now carried by
  // the synthesized state machine and override mux, so the rwprobe result type
  // must not leak into the output.
  for (auto fop : forceables)
    firrtl::detail::replaceWithNewForceability(fop, false);

  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Unhandled
//===----------------------------------------------------------------------===//

LogicalResult ProbeVisitor::visitUnhandledOp(Operation *op) {
  auto checkType = [&](auto type) -> bool {
    // Return if conversion needed (or if error).
    auto newType = convertType(type, op->getLoc());
    if (failed(newType))
      return true;
    if (!*newType)
      return false;

    // Type found that needs to be converted, diagnose.
    op->emitError("unhandled operation needs conversion of type ")
        << type << " to " << *newType;
    return true;
  };

  return success(llvm::none_of(op->getOperandTypes(), checkType) &&
                 llvm::none_of(op->getResultTypes(), checkType));
}

//===----------------------------------------------------------------------===//
// Visitor: CHIRRTL
//===----------------------------------------------------------------------===//
LogicalResult
ProbeVisitor::visitMemoryDebugPortOp(chirrtl::MemoryDebugPortOp op) {
  auto conv = convertType(op.getResult().getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto type = *conv;
  assert(type);

  auto vectype = type_cast<FVectorType>(type);

  // Just assert the chirrtl memory IR has the expected structure,
  // if it didn't many things break.
  // Must be defined in same module, tapped memory must be comb mem.
  auto mem = op.getMemory().getDefiningOp<chirrtl::CombMemOp>();
  assert(mem);

  // The following is adapted from LowerAnnotations.
  Value clock;
  for (auto *portOp : mem.getResult().getUsers()) {
    for (auto result : portOp->getResults()) {
      for (auto *user : result.getUsers()) {
        auto accessOp = dyn_cast<chirrtl::MemoryPortAccessOp>(user);
        if (!accessOp)
          continue;
        auto newClock = accessOp.getClock();
        if (clock && clock != newClock)
          return mem.emitOpError(
              "has different clocks on different ports (this is ambiguous "
              "when compiling without reference types)");
        clock = newClock;
      }
    }
  }
  if (!clock)
    return mem->emitOpError(
        "does not have an access port to determine a clock connection (this "
        "is necessary when compiling without reference types)");

  // Add one port per memory address.
  SmallVector<Value> data;
  ImplicitLocOpBuilder builder(op.getLoc(), op);

  // Insert new ports as late as possible (end of block containing the memory).
  // This is necessary to preserve ordering of existing ports.
  builder.setInsertionPointToEnd(mem->getBlock());
  Type uintType = builder.getType<UIntType>();
  for (uint64_t i = 0, e = mem.getType().getNumElements(); i != e; ++i) {
    auto port = chirrtl::MemoryPortOp::create(
        builder, mem.getType().getElementType(),
        chirrtl::CMemoryPortType::get(builder.getContext()), mem.getResult(),
        MemDirAttr::Read, builder.getStringAttr("memTap_" + Twine(i)),
        builder.getArrayAttr({}));
    chirrtl::MemoryPortAccessOp::create(
        builder, port.getPort(),
        ConstantOp::create(builder, uintType, APSInt::getUnsigned(i)), clock);
    data.push_back(port.getData());
  }

  // Package up all the reads into a vector.
  assert(vectype == FVectorType::get(mem.getType().getElementType(),
                                     mem.getType().getNumElements()));
  auto vecData = VectorCreateOp::create(builder, vectype, data);

  // While the new ports are added as late as possible, the debug port
  // operation we're replacing likely has users and those are before
  // the new ports.  Add a wire at a point we know dominates this operation
  // and the new port access operations added above.  This will be used for
  // the existing users of the debug port.
  builder.setInsertionPoint(mem);
  auto wire = WireOp::create(builder, vectype);
  builder.setInsertionPointToEnd(mem->getBlock());
  emitConnect(builder, wire.getData(), vecData);
  probeToHWMap[op.getResult()] = wire.getData();
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Declarations
//===----------------------------------------------------------------------===//

LogicalResult ProbeVisitor::visitDecl(MemOp op) {
  // Scan for debug ports.  These are not supported presently, diagnose.
  SmallVector<Type> newTypes;
  auto needsConv = mapRange(op->getResultTypes(), op->getLoc(), newTypes);
  if (failed(needsConv))
    return failure();
  if (!*needsConv)
    return success();

  return op.emitError("memory has unsupported debug port (memtap)");
}

LogicalResult ProbeVisitor::visitDecl(WireOp op) {
  if (op.isForceable())
    return visitActiveForceableDecl(op);

  auto conv = convertType(op.getDataRaw().getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto type = *conv;
  if (!type) // No conversion needed.
    return success();

  // New Wire of converted type.
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto cloned = cast<WireOp>(builder.clone(*op));
  cloned->getOpResults().front().setType(type);
  probeToHWMap[op.getDataRaw()] = cloned.getData();
  toDelete.push_back(op);
  return success();
}
/// Subfield-extract one named element from a bundle-typed value.
static Value getBundleField(ImplicitLocOpBuilder &builder, Value bundle,
                            StringRef fieldName) {
  auto bundleType = type_cast<BundleType>(bundle.getType());
  auto idx = bundleType.getElementIndex(fieldName);
  assert(idx && "field not found in bundle");
  return SubfieldOp::create(builder, bundle, *idx);
}

/// The four fields of a `_force_ctrl` control bundle, read or to-be-connected
/// as SSA values.  Centralizes the repeated field access for the cross-module
/// control path.
struct ForceCtrlFields {
  Value forceActive;
  Value releaseActive;
  Value forcedValue;
  Value clk;
};

/// Read all four fields of a `_force_ctrl` control bundle into a
/// `ForceCtrlFields`.
static ForceCtrlFields readForceCtrlFields(ImplicitLocOpBuilder &builder,
                                           Value bundle) {
  return {getBundleField(builder, bundle, "forceActive"),
          getBundleField(builder, bundle, "releaseActive"),
          getBundleField(builder, bundle, "forcedValue"),
          getBundleField(builder, bundle, "clk")};
}

/// Create a 1-bit UInt constant (0 or 1).
static Value createU1Const(ImplicitLocOpBuilder &builder, bool value) {
  return builder.createOrFold<ConstantOp>(
      APSInt(APInt(1, value ? 1 : 0, /*isSigned=*/false), /*isUnsigned=*/true));
}

LogicalResult ProbeVisitor::visitActiveForceableDecl(Forceable fop) {
  assert(fop.isForceable() && "must be called on active forceables");
  // Map rw ref result to normal result.
  auto data = fop.getData();
  auto conv = mapType(fop.getDataRef().getType(), fop.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;
  forceables.push_back(fop);

  assert(newType == data.getType().getPassiveType());
  if (newType != data.getType()) {
    ImplicitLocOpBuilder builder(fop.getLoc(), fop);
    builder.setInsertionPointAfterValue(data);
    auto wire = WireOp::create(builder, newType);
    emitConnect(builder, wire.getData(), data);
    data = wire.getData();
  }
  probeToHWMap[fop.getDataRef()] = data;

  // The force/release state machine is synthesized lazily, only if this target
  // is actually forced/released (locally via forceReleaseMap, or from outside
  // via a forceable port).
  return success();
}

WireOp ProbeVisitor::synthesizeForceableStateMachine(FIRRTLBaseType probedType,
                                                     Value data) {
  Location loc = data.getLoc();

  ImplicitLocOpBuilder builder(loc, data.getContext());

  auto bundleType = createForceCtrlBundleType(probedType);

  auto fModule = getParentModule(data);
  assert(fModule && "Expected to find parent FModuleOp");

  builder.setInsertionPointToStart(fModule.getBodyBlock());
  auto forceCntrlWire = WireOp::create(builder, bundleType);
  // The state-machine registers read their control inputs from the bundle wire
  // fields; the wire itself is driven later (once) by the sequential post-pass.
  // Cross-module targets (exported ports / instance forwarding) need this wire;
  // purely-local targets use buildStateMachineRegisters directly and create no
  // wire.
  StateMachineInputs in;
  auto fields = readForceCtrlFields(builder, forceCntrlWire.getData());
  in.forceActive = fields.forceActive;
  in.releaseActive = fields.releaseActive;
  in.forcedValue = fields.forcedValue;
  in.clk = fields.clk;

  if (failed(buildStateMachineRegisters(probedType, data, in)))
    return {};

  return forceCntrlWire;
}

LogicalResult
ProbeVisitor::buildStateMachineRegisters(FIRRTLBaseType probedType, Value data,
                                         const StateMachineInputs &in) {
  Location loc = data.getLoc();
  ImplicitLocOpBuilder builder(loc, data.getContext());

  auto fModule = getParentModule(data);
  assert(fModule && "Expected to find parent FModuleOp");

  Value forceActive = in.forceActive;
  Value releaseActive = in.releaseActive;
  Value forcedValue = in.forcedValue;
  Value fClk = in.clk;
  if (!fClk)
    return mlir::emitError(loc, "cannot synthesize force/release: no clock "
                                "available (all accesses are `_initial`)");

  /*
  when (forceActive) {
    forced := true.B
    forcedValue := forceValue
  }.elsewhen(releaseActive) {
    forced := false.B
  }

  when (forced) {
    probe := forcedValue
  }
  */
  if (auto *defOp = data.getDefiningOp()) {
    builder.setInsertionPointAfter(defOp);
  } else {
    auto blockArg = cast<BlockArgument>(data);
    builder.setInsertionPointToStart(blockArg.getOwner());
  }

  // Look for a reset port so the synthesized registers start at 0 rather than
  // X in simulation.  Only ground types with a known bit-width can produce a
  // typed reset-value constant; fall back to no-reset otherwise.
  // RegResetOp accepts AnyResetType directly (sync or async), so no cast is
  // needed — the FIRRTL type encodes the reset flavour.
  Value resetSig;
  int64_t probedWidth = probedType.getBitWidthOrSentinel();
  for (auto [i, portAttr] : llvm::enumerate(fModule.getPortNamesAttr())) {
    if (cast<StringAttr>(portAttr).getValue() == "reset") {
      resetSig = fModule.getBodyBlock()->getArgument(i);
      break;
    }
  }

  // Create the FIRRTL registers first so their results can be referenced
  // directly in the next-state muxes below.  FIRRTL registers are driven by
  // connects rather than SSA next-state operands, so self-reference is legal.
  Value forcedReg, forcedValueReg;
  if (resetSig && probedWidth >= 0) {
    Value resetValI1 = createU1Const(builder, false);
    Value resetValData = builder.createOrFold<ConstantOp>(
        APSInt(APInt(probedWidth, 0, /*isSigned=*/false), /*isUnsigned=*/true));
    forcedReg = RegResetOp::create(builder, u1Type, fClk, resetSig, resetValI1,
                                   "forced")
                    .getResult();
    forcedValueReg = RegResetOp::create(builder, probedType, fClk, resetSig,
                                        resetValData, "forcedValue")
                         .getResult();
  } else {
    forcedReg = RegOp::create(builder, u1Type, fClk, "forced").getResult();
    forcedValueReg =
        RegOp::create(builder, probedType, fClk, "forcedValue").getResult();
  }

  // Build next-state muxes referencing the register results directly.  The
  // reduced control values (forceActive/releaseActive/forcedValue) for a
  // purely-local target are computed at the end of the block, so build the
  // next-state muxes and their feedback connects there to keep SSA dominance.
  builder.setInsertionPointToEnd(fModule.getBodyBlock());
  Value cZero = createU1Const(builder, false);
  Value cOne = createU1Const(builder, true);
  Value forcedNext = builder.createOrFold<MuxPrimOp>(
      forceActive, cOne,
      builder.createOrFold<MuxPrimOp>(releaseActive, cZero, forcedReg));
  Value forcedValueNext =
      builder.createOrFold<MuxPrimOp>(forceActive, forcedValue, forcedValueReg);

  // Close the feedback loop: connect next-state muxes back into the registers.
  builder.create<MatchingConnectOp>(forcedReg, forcedNext);
  builder.create<MatchingConnectOp>(forcedValueReg, forcedValueNext);

  Value defaultSrc;
  Operation *existingConnect = nullptr;

  auto target = data;
  if (foldFlow(target) == Flow::Source) {
    mlir::emitError(loc, "cannot synthesize force/release: target is read-only "
                         "(source flow) and cannot be driven");
    return failure();
  }

  for (auto &use : target.getUses()) {
    auto fconn = dyn_cast<FConnectLike>(use.getOwner());
    if (fconn && fconn.getDest() == target) {
      existingConnect = fconn;
      defaultSrc = fconn.getSrc();
      break;
    }
  }

  if (!defaultSrc)
    defaultSrc = builder.createOrFold<InvalidValueOp>(probedType);
  if (existingConnect)
    existingConnect->erase();

  builder.setInsertionPointToEnd(fModule.getBodyBlock());
  MatchingConnectOp::create(
      builder, data,
      builder.createOrFold<MuxPrimOp>(forcedReg, forcedValueReg, defaultSrc));

  return success();
}

LogicalResult ProbeVisitor::visitInstanceLike(FInstanceLike oldInst) {
  SmallVector<Type> newTypes;
  auto needsConv =
      mapRange(oldInst->getResultTypes(), oldInst->getLoc(), newTypes);
  if (failed(needsConv))
    return failure();

  if (!*needsConv)
    return success();

  // Rebuild the instance with the probe result types converted in place.  This
  // is purely module-local and parallel-safe.  Adding the `_force_ctrl` ports
  // for forceable probe ports is handled by a sequential post-pass in
  // `runOnOperation` (so a module's port count and its instances' port counts
  // are mutated together and stay in lockstep).
  ImplicitLocOpBuilder builder(oldInst->getLoc(), oldInst);
  auto *newInst = builder.clone(*oldInst);
  for (auto [idx, oldNewType] : llvm::enumerate(llvm::zip_equal(
           oldInst->getOpResults(), newInst->getOpResults(), newTypes))) {
    auto [oldResult, newResult, newType] = oldNewType;
    if (newType == oldResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }

    newResult.setType(newType);
    probeToHWMap[oldResult] = newResult;

    // For a forceable probe result, create a module-local forwarding control
    // wire now (so the parent's force/release reduction can target it like any
    // other control wire).  Defer connecting the instance's `_force_ctrl` input
    // port to it until the post-pass inserts that port.
    auto refType = dyn_cast<RefType>(oldResult.getType());
    if (refType && refType.getForceable()) {
      auto probedType = type_cast<FIRRTLBaseType>(refType.getType());
      auto bundleType = createForceCtrlBundleType(probedType);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(
          oldInst->getParentOfType<FModuleOp>().getBodyBlock());
      auto forwardingWire = WireOp::create(builder, bundleType);
      // Key by the converted hw result (newResult), consistent with all other
      // probeToHWMap entries which use the hardware value as the key.
      targetToForceCtrlWire[newResult] = forwardingWire;
      instancePlumb.push_back(
          {newInst, (unsigned)idx, forwardingWire.getData()});

      // Check if oldResult is used for forcing: either through an output port
      // (via RefDefineOp to a BlockArgument output) or by force/release ops.
      bool usedForForcing = false;
      for (auto *user : oldResult.getUsers()) {
        // Check for force/release operations
        if (isa<RefForceOp, RefReleaseOp, RefForceInitialOp,
                RefReleaseInitialOp>(user)) {
          usedForForcing = true;
          break;
        }
        // Check for RefDefineOp that connects to an output port
        if (auto refDefine = dyn_cast<RefDefineOp>(user)) {
          auto dest = refDefine.getDest();
          if (auto blockArg = dyn_cast<BlockArgument>(dest)) {
            // Check if this is an output port
            auto parentModule =
                dyn_cast<FModuleOp>(blockArg.getOwner()->getParentOp());
            if (parentModule &&
                parentModule.getPortDirection(blockArg.getArgNumber()) ==
                    Direction::Out) {
              usedForForcing = true;
              break;
            }
          }
        }
      }

      // If not used for forcing, initialize the control wire with default
      // values: forceActive = false, releaseActive = false, forcedValue =
      // invalid
      if (!usedForForcing) {
        ImplicitLocOpBuilder ctrlBuilder(forwardingWire.getLoc(),
                                         forwardingWire);
        ctrlBuilder.setInsertionPointAfter(forwardingWire);
        auto fields =
            readForceCtrlFields(ctrlBuilder, forwardingWire.getData());
        Value falseVal = createU1Const(ctrlBuilder, false);
        Value invalidVal = ctrlBuilder.createOrFold<InvalidValueOp>(probedType);
        // Note: clock field doesn't need initialization as it will be driven
        // by the post-pass if needed
        ctrlBuilder.create<MatchingConnectOp>(fields.forceActive, falseVal);
        ctrlBuilder.create<MatchingConnectOp>(fields.releaseActive, falseVal);
        ctrlBuilder.create<MatchingConnectOp>(fields.forcedValue, invalidVal);
      }
    }
  }

  toDelete.push_back(oldInst);
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Probe operations
//===----------------------------------------------------------------------===//

LogicalResult ProbeVisitor::visitStmt(RefDefineOp op) {
  // ref.define x, y -> connect map(x), map(y)
  // Be mindful of connect semantics when considering
  // placement.
  auto newDest = probeToHWMap.at(op.getDest());
  auto newSrc = probeToHWMap.at(op.getSrc());

  // Source must be ancestor of destination block for a connect
  // to behave the same (generally).
  assert(!isa<BlockArgument>(newDest));
  auto *destDefiningOp = newDest.getDefiningOp();
  assert(destDefiningOp);
  if (!newSrc.getParentBlock()->findAncestorOpInBlock(*destDefiningOp)) {
    // Conditional or sending out of a layer...
    auto diag = op.emitError("unable to convert to equivalent connect");
    diag.attachNote(op.getDest().getLoc()) << "destination here";
    diag.attachNote(op.getSrc().getLoc()) << "source here";
    return diag;
  }

  auto *destBlock = newDest.getParentBlock();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(op.getLoc(), destBlock);
  emitConnect(builder, newDest, newSrc);
  toDelete.push_back(op);

  return success();
}

LogicalResult ProbeVisitor::visitExpr(RWProbeOp op) {
  // Handle similar to ref.send but lookup the target
  // and materialize a value for it (indexing).
  auto conv = mapType(op.getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;
  toDelete.push_back(op);

  auto ist = irn.lookup(op.getTarget());
  assert(ist);
  auto ref = getFieldRefForTarget(ist);

  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(ref.getValue());
  auto data = getValueByFieldID(builder, ref.getValue(), ref.getFieldID());
  assert(cast<FIRRTLBaseType>(data.getType()).getPassiveType() ==
         op.getType().getType());
  if (newType != data.getType()) {
    auto wire = WireOp::create(builder, newType);
    emitConnect(builder, wire.getData(), data);
    data = wire.getData();
  }
  probeToHWMap[op.getResult()] = data;

  // The force/release state machine for a forceable target is synthesized
  // lazily (only when the probe is actually forced/released); here we only
  // record the hardware value.
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefCastOp op) {
  auto input = probeToHWMap.at(op.getInput());
  // Insert wire of the new type, and connect to it.

  // y = ref.cast x : probe<t1> -> probe<t2>
  // ->
  // w = firrtl.wire : t2
  // emitConnect(w : t2, map(x): t1)

  auto conv = mapType(op.getResult().getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;

  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(input);
  auto wire = WireOp::create(builder, newType);
  emitConnect(builder, wire.getData(), input);
  probeToHWMap[op.getResult()] = wire.getData();
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefSendOp op) {
  auto conv = mapType(op.getResult().getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;
  toDelete.push_back(op);

  // If the mapped type is same as input, just use that.
  if (newType == op.getBase().getType()) {
    probeToHWMap[op.getResult()] = op.getBase();
    return success();
  }

  // Otherwise, need to make this the probed type (passive).
  // Insert wire of the new type, and connect to it.
  assert(newType == op.getBase().getType().getPassiveType());
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(op.getBase());
  auto wire = WireOp::create(builder, newType);
  emitConnect(builder, wire.getData(), op.getBase());
  probeToHWMap[op.getResult()] = wire.getData();
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefResolveOp op) {
  // ref.resolve x -> map(x)
  auto val = probeToHWMap.at(op.getRef());
  op.replaceAllUsesWith(val);
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefSubOp op) {
  // ref.sub x, fieldid -> index(map(x), fieldid)
  auto val = probeToHWMap.at(op.getInput());
  assert(val);
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(op.getInput());
  auto newVal =
      getValueByFieldID(builder, val, op.getAccessedField().getFieldID());
  probeToHWMap[op.getResult()] = newVal;
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Force/Release Synthesis
//===----------------------------------------------------------------------===//

/// Combine a local-reduced (lF, lR, lV) tuple with an inbound `_force_ctrl`
/// bundle (iF, iR, iV) into a single (oF, oR, oV) tuple. Local force wins
/// over inbound when both fire simultaneously.  A purely-inbound target (no
/// local force/release) passes null local values; those are treated as
/// constant 0 / invalid, so the OR/mux fold back to the inbound fields — the
/// same one code path serves both the local+inbound and inbound-only cases.
static void
combineWithInboundCtrl(ImplicitLocOpBuilder &builder, Value localForceActive,
                       Value localReleaseActive, Value localForceValue,
                       FIRRTLBaseType probedType, Value inboundBundle,
                       Value &outForceActive, Value &outReleaseActive,
                       Value &outForceValue) {
  if (!localForceActive)
    localForceActive = createU1Const(builder, false);
  if (!localReleaseActive)
    localReleaseActive = createU1Const(builder, false);
  if (!localForceValue)
    localForceValue = builder.createOrFold<InvalidValueOp>(probedType);
  Value iF = getBundleField(builder, inboundBundle, "forceActive");
  Value iR = getBundleField(builder, inboundBundle, "releaseActive");
  Value iV = getBundleField(builder, inboundBundle, "forcedValue");
  outForceActive = builder.createOrFold<OrPrimOp>(localForceActive, iF);
  outReleaseActive = builder.createOrFold<OrPrimOp>(localReleaseActive, iR);
  outForceValue =
      builder.createOrFold<MuxPrimOp>(localForceActive, localForceValue, iV);
}

/// Connect the four fields of a `_force_ctrl` control-bundle wire exactly once
/// from already-computed field values.  Emits a diagnostic and fails if `clk`
/// is null (every contributing access was an `_initial` variant, so the
/// synthesized registers would have no clock to run on).
static LogicalResult connectControlWireFields(ImplicitLocOpBuilder &builder,
                                              WireOp controlWire,
                                              Value forceActive,
                                              Value releaseActive,
                                              Value forcedValue, Value clk) {

  if (!clk)
    return mlir::emitError(controlWire.getLoc(),
                           "cannot synthesize force/release: no clock "
                           "available (all accesses are `_initial`)");
  auto dst = readForceCtrlFields(builder, controlWire.getData());
  builder.createOrFold<MatchingConnectOp>(dst.forceActive, forceActive);
  builder.createOrFold<MatchingConnectOp>(dst.releaseActive, releaseActive);
  builder.createOrFold<MatchingConnectOp>(dst.forcedValue, forcedValue);
  builder.createOrFold<MatchingConnectOp>(dst.clk, clk);
  return success();
}

/// Reduce the priority-ordered force/release accesses for one target into a
/// single `LocalCtrl`.  See the declaration for insertion-point contract.
ProbeVisitor::LocalCtrl
ProbeVisitor::reduceAccesses(ImplicitLocOpBuilder &builder,
                             FIRRTLBaseType probedType,
                             ArrayRef<ForceReleaseAccess> accesses) {
  Value cZero = createU1Const(builder, false);
  Value cOne = createU1Const(builder, true);

  Value forceWins = cZero;
  Value forceValue = builder.createOrFold<InvalidValueOp>(probedType);
  SmallVector<ForceReleaseAccess> forces, releases;
  Value freeRunningClock = {};
  for (auto &access : accesses) {
    Value isForceVal = access.isForce() ? cOne : cZero;
    builder.setInsertionPointAfter(access.op);
    forceWins = builder.createOrFold<MuxPrimOp>(access.predicate, isForceVal,
                                                forceWins);
    if (!freeRunningClock && access.clock)
      freeRunningClock = access.clock;
    if (access.isForce()) {
      forceValue = builder.createOrFold<MuxPrimOp>(
          access.predicate, access.forceValue.value(), forceValue);
      forces.push_back(access);
    } else
      releases.push_back(access);
  }
  builder.setInsertionPointToEnd(builder.getInsertionBlock());

  // Starting from the first predicate (rather than a cZero seed) let the
  // single-predicate case fold to just the predicate value.
  auto orReduce = [&](ArrayRef<ForceReleaseAccess> set) -> Value {
    if (set.empty())
      return cZero;
    Value v = set.front().predicate;
    for (auto &a : set.drop_front())
      v = builder.createOrFold<OrPrimOp>(v, a.predicate);
    return v;
  };

  // Gate force predicates with forceWins to avoid overwriting forcedValue on
  // a cycle where a concurrent higher-priority release fires.
  Value forceActive =
      forces.empty()
          ? Value(cZero)
          : builder.createOrFold<AndPrimOp>(orReduce(forces), forceWins);

  // Only clear the forced flag when no higher-priority force fires
  // simultaneously.
  Value releaseActive =
      releases.empty()
          ? Value(cZero)
          : builder.createOrFold<AndPrimOp>(
                orReduce(releases), builder.createOrFold<NotPrimOp>(forceWins));

  return {forceActive, releaseActive, forceValue, freeRunningClock};
}

/// Reduce every target's accesses to a `LocalCtrl`.  Makes no materialization
/// decision.
LogicalResult ProbeVisitor::reduceForceReleaseControl(FModuleLike mod) {
  auto *block = getBodyBlock(mod);
  if (!block || forceReleaseMap.empty())
    return success();

  for (auto &[hwVal, accesses] : forceReleaseMap) {
    if (accesses.empty())
      continue;

    // The forceReleaseMap is keyed by the canonical hardware value (not a
    // probe SSA value), so both force and release ops targeting the same wire
    // land in the same entry regardless of which firrtl.ref.rwprobe SSA value
    // they used as their dest operand.
    auto probedType = type_cast<FIRRTLBaseType>(hwVal.getType());

    Location loc = accesses[0].op->getLoc();
    ImplicitLocOpBuilder builder(loc, mod);
    builder.setInsertionPointToStart(block);

    // Materialization (registers / wires / muxes) is deferred to
    // `materializeForcePlans`, which runs after `analyzeForcePlans` has
    // classified each target.  Here we only compute and stash the reduced
    // control.
    localCtrlMap[hwVal] = reduceAccesses(builder, probedType, accesses);
  }

  return success();
}

/// Classify each forced target into a `ForcePlan`.
void ProbeVisitor::analyzeForcePlans() {
  // Iterate in forceReleaseMap (MapVector) order for determinism.
  for (auto &kv : forceReleaseMap) {
    Value target = kv.first;
    auto localIt = localCtrlMap.find(target);
    if (localIt == localCtrlMap.end())
      continue;

    ForcePlan plan;
    plan.target = target;
    plan.probedType = type_cast<FIRRTLBaseType>(target.getType());
    plan.local = localIt->second;

    if (exportedProbes.contains(target)) {
      // Driven by the sequential post-pass (merges local + inbound); recorded
      // here only for completeness / determinism.
      plan.category = ForceCategory::Exported;
    } else if (auto wireIt = targetToForceCtrlWire.find(target);
               wireIt != targetToForceCtrlWire.end()) {
      // The real SM lives in a child instance; drive the parent's forwarding
      // control-bundle wire.
      plan.category = ForceCategory::InstanceForwarded;
      plan.forwardingWire = wireIt->second;
    } else {
      // Purely local: registers driven directly from the reduced SSA values.
      plan.category = ForceCategory::Local;
    }

    forcePlans.push_back(plan);
  }
}

/// Realize each `ForcePlan` with a single dispatch on its category.
LogicalResult ProbeVisitor::materializeForcePlans() {
  for (auto &plan : forcePlans) {
    auto &l = plan.local;
    switch (plan.category) {
    case ForceCategory::Exported:
      // Driven by the sequential post-pass (merging local + inbound) so each
      // control-wire field keeps exactly one driver.
      continue;

    case ForceCategory::InstanceForwarded: {
      WireOp forceCntrlWire = plan.forwardingWire;
      auto *block = forceCntrlWire->getBlock();
      ImplicitLocOpBuilder builder(forceCntrlWire.getLoc(), block,
                                   block->end());
      if (failed(connectControlWireFields(builder, forceCntrlWire,
                                          l.forceActive, l.releaseActive,
                                          l.forcedValue, l.clk)))
        return failure();
      continue;
    }

    case ForceCategory::Local:
      if (failed(buildStateMachineRegisters(
              plan.probedType, plan.target,
              {l.forceActive, l.releaseActive, l.forcedValue, l.clk})))
        return failure();
      continue;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Force/Release operations
//===----------------------------------------------------------------------===//

// Collected access ops are held in `forceReleaseMap` until
// `reduceForceReleaseControl` reduces them to a `LocalCtrl` and
// `materializeForcePlans` builds the per-probe state machine, at which point
// the originals are erased.  `_initial` variants carry a null clock — they
// have no synchronous component and ride along on a sibling clocked
// access's state machine.

LogicalResult ProbeVisitor::visitStmt(RefForceOp op) {
  Value hwDest = probeToHWMap.lookup(op.getDest());
  assert(hwDest && "forced probe has no hardware mapping");
  forceReleaseMap[hwDest].push_back(
      {op, op.getPredicate(), op.getSrc(), op.getClock()});
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitStmt(RefForceInitialOp op) {
  Value hwDest = probeToHWMap.lookup(op.getDest());
  assert(hwDest && "forced probe has no hardware mapping");
  forceReleaseMap[hwDest].push_back(
      {op, op.getPredicate(), op.getSrc(), /*clock=*/Value()});
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitStmt(RefReleaseOp op) {
  Value hwDest = probeToHWMap.lookup(op.getDest());
  assert(hwDest && "released probe has no hardware mapping");
  forceReleaseMap[hwDest].push_back(
      {op, op.getPredicate(), std::nullopt, op.getClock()});
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitStmt(RefReleaseInitialOp op) {
  Value hwDest = probeToHWMap.lookup(op.getDest());
  assert(hwDest && "released probe has no hardware mapping");
  forceReleaseMap[hwDest].push_back(
      {op, op.getPredicate(), std::nullopt, /*clock=*/Value()});
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct ProbesToSignalsPass
    : public circt::firrtl::impl::ProbesToSignalsBase<ProbesToSignalsPass> {
  ProbesToSignalsPass() = default;
  void runOnOperation() override;
};
} // end anonymous namespace

void ProbesToSignalsPass::runOnOperation() {
  CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);

  // Gated-clock conversion is demand-driven: roots are added by the
  // per-module visitor, then `tracer.run()` is invoked at the end of each
  // module's visit to plumb (base, enable) port pairs across module
  // boundaries.  Sequential / not thread-safe.
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  GatedClockConversion tracer(instanceGraph);

  // `_initial` force/release variants carry no clock, so they are not
  // gated-clock roots; they ride along on a sibling clocked access's state
  // machine (or are diagnosed if no clocked access exists).
  getOperation()->walk([&](Operation *op) {
    if (isa<RefForceOp, RefReleaseOp>(op)) {
      if (failed(tracer.addRoot(op)))
        return signalPassFailure();
    } else if (auto regOp = dyn_cast<RegOp>(op)) {
      if (regOp.isForceable())
        if (failed(tracer.addRoot(op)))
          return signalPassFailure();
    } else if (auto regResetOp = dyn_cast<RegResetOp>(op)) {
      if (regResetOp.isForceable())
        if (failed(tracer.addRoot(op)))
          return signalPassFailure();
    }
  });
  if (failed(tracer.run()))
    return signalPassFailure();

  hw::InnerRefNamespace irn{getAnalysis<SymbolTable>(),
                            getAnalysis<hw::InnerSymbolTableCollection>()};

  // Per-module deferred force-control plumbing, collected during the parallel
  // phase and replayed sequentially below.  Each slot is written by exactly one
  // thread (its own index), so this is race-free.
  struct ModuleWork {
    FModuleLike mod;
    SmallVector<ProbeVisitor::ExportEntry> exportWork;
    SmallVector<ProbeVisitor::InstancePlumb> instancePlumb;
    ModuleWork(FModuleLike mod) : mod(mod) {}
  };

  SmallVector<ModuleWork> deferred = llvm::to_vector(
      llvm::map_range(getOperation().getOps<FModuleLike>(),
                      [&](FModuleLike mod) { return ModuleWork(mod); }));

  auto result =
      failableParallelForEach(&getContext(), deferred, [&](ModuleWork &w) {
        ProbeVisitor visitor(irn);
        auto mod = w.mod;
        if (failed(visitor.visit(mod)))
          return failure();
        w.exportWork = std::move(visitor.exportWork);
        w.instancePlumb = std::move(visitor.instancePlumb);
        return success();
      });

  if (result.failed())
    return signalPassFailure();

  // Sequential post-pass: mutate module signatures by inserting the inbound
  // `<name>_force_ctrl` ports.

  // (live instance op, forceable result idx) -> parent forwarding control wire.
  DenseMap<std::pair<Operation *, unsigned>, Value> instForwarding;
  for (auto &w : deferred)
    for (auto &p : w.instancePlumb)
      instForwarding[{p.inst, p.resultIdx}] = p.forwardingWire;

  // Fresh instance graph: the parallel phase rebuilt instances, so the cached
  // analysis is stale.
  InstanceGraph postIG(getOperation());

  SmallVector<Operation *> instToErase;
  for (auto &w : deferred) {
    if (w.exportWork.empty())
      continue;
    auto mod = w.mod;
    auto *ctx = mod->getContext();

    // Build the new input ports, all appended at the end.
    unsigned appendAt = mod.getNumPorts();
    SmallVector<std::pair<unsigned, PortInfo>> newPorts;
    newPorts.reserve(w.exportWork.size());
    for (auto &e : w.exportWork) {
      auto ctrlName =
          StringAttr::get(ctx, e.portName.getValue().str() + "_force_ctrl");
      newPorts.emplace_back(
          appendAt,
          PortInfo(ctrlName, e.controlWire.getData().getType(), Direction::In,
                   /*symName=*/StringAttr{}, e.portLoc));
    }
    mod.insertPorts(newPorts);

    // Drive each exported probe's control wire once.  Local control is treated
    // uniformly as an OR-input to the inbound bundle: an all-null LocalCtrl (no
    // local force) folds back to the plain inbound fields, so there is a single
    // code path for both the local+inbound and inbound-only cases.
    if (auto fmod = dyn_cast<FModuleOp>(*mod)) {
      auto *block = fmod.getBodyBlock();
      for (auto [j, e] : llvm::enumerate(w.exportWork)) {
        Value inbound = block->getArgument(appendAt + j);
        ImplicitLocOpBuilder builder(e.portLoc, block, block->end());
        Value fa, ra, fv;
        combineWithInboundCtrl(builder, e.local.forceActive,
                               e.local.releaseActive, e.local.forcedValue,
                               e.probedType, inbound, fa, ra, fv);
        // The clock comes from local control when present, otherwise from the
        // inbound bundle (an exported wire target has no local clock).
        Value clk =
            e.local.clk ? e.local.clk : getBundleField(builder, inbound, "clk");
        if (failed(connectControlWireFields(builder, e.controlWire, fa, ra, fv,
                                            clk)))
          return signalPassFailure();
      }
    }

    // Insert matching ports on every instance of `mod` and wire each new
    // `_force_ctrl` input to the parent's forwarding control wire.
    auto *node = postIG.lookup(mod);
    SmallVector<FInstanceLike> oldInsts;
    for (auto *use : node->uses())
      oldInsts.push_back(use->getInstance<FInstanceLike>());
    for (auto oldInst : oldInsts) {
      unsigned origCount = oldInst->getNumResults();
      auto newInst = oldInst.cloneWithInsertedPortsAndReplaceUses(newPorts);
      postIG.replaceInstance(oldInst, newInst);

      ImplicitLocOpBuilder builder(newInst->getLoc(), newInst);
      builder.setInsertionPointAfter(newInst);
      for (auto [j, e] : llvm::enumerate(w.exportWork)) {
        auto it = instForwarding.find({oldInst.getOperation(), e.portIdx});
        if (it == instForwarding.end())
          continue;
        MatchingConnectOp::create(builder, newInst->getResult(origCount + j),
                                  it->second);
      }
      instToErase.push_back(oldInst);
    }
  }

  for (auto *op : instToErase)
    op->erase();
}
