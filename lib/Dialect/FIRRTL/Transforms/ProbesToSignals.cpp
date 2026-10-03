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
// Force/release on RWProbes is synthesized per target. Forceable probe ports
// gain an appended input carrying the force control:
//
//   probe<T>              -> T
//   rwprobe<T>, port `p`  -> out `p`: T, plus
//                            in `p_force_ctrl`:
//                              { forceActive, releaseActive, forcedValue, clk }
//
// Control ports are appended, preserving original port indices.
//
// Only the event is sampled; the winning force's RHS stays live, matching
// Verilog `force a = v`. Overrides are injected on redirectable reads so the
// target remains single-driven. Multiple probe ports share one target state
// machine; later ports have priority, and local control has priority over port
// control.
//
// Gated clocks are converted first so synthesized state uses a free-running
// clock.
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
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/Debug.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

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

FModuleOp getParentModule(Value value) {
  if (isa<BlockArgument>(value))
    return cast<FModuleOp>(value.getParentBlock()->getParentOp());
  return value.getDefiningOp()->getParentOfType<FModuleOp>();
}

Value getBundleField(ImplicitLocOpBuilder &builder, Value bundle,
                     StringRef fieldName) {
  auto bundleType = type_cast<BundleType>(bundle.getType());
  auto idx = bundleType.getElementIndex(fieldName);
  assert(idx && "field not found in bundle");
  return SubfieldOp::create(builder, bundle, *idx);
}

struct ForceReleaseAccess {
  Operation *op;
  Value predicate;
  std::optional<Value> forceValue;
  Value clock;

  bool isForce() const { return forceValue.has_value(); }
};

/// A null field means that local control is absent.
struct CtrlGroup {
  Value forceActive;
  Value releaseActive;
  Value forcedValue;
};

/// Reduced control and its clock.
struct ForceCtrl {
  CtrlGroup group;
  Value clk;
};

/// Build a `UInt<1>` constant at the builder's insertion point. Keep constants
/// local because control may be materialized in a nested region.
Value getU1Const(ImplicitLocOpBuilder &builder, bool value) {
  return ConstantOp::create(builder, APSInt(APInt(1, value ? 1 : 0,
                                                  /*isSigned=*/false),
                                            /*isUnsigned=*/true));
}

BundleType createForceCtrlBundleType(FIRRTLBaseType probedType) {
  auto *ctx = probedType.getContext();
  auto u1Type = UIntType::get(ctx, 1);
  auto clkType = ClockType::get(ctx);
  SmallVector<BundleType::BundleElement> elements = {
      {StringAttr::get(ctx, "forceActive"), /*isFlip=*/false, u1Type},
      {StringAttr::get(ctx, "releaseActive"), /*isFlip=*/false, u1Type},
      {StringAttr::get(ctx, "forcedValue"), /*isFlip=*/false, probedType},
      {StringAttr::get(ctx, "clk"), /*isFlip=*/false, clkType},
  };
  return BundleType::get(ctx, elements);
}

class ProbeVisitor : public FIRRTLVisitor<ProbeVisitor, LogicalResult> {
public:
  static constexpr StringRef forceActiveName = "forceActive";
  static constexpr StringRef releaseActiveName = "releaseActive";
  static constexpr StringRef forcedValueName = "forcedValue";
  static constexpr StringRef clockName = "clk";

  ProbeVisitor(hw::InnerRefNamespace &irn, InstanceGraph &instanceGraph)
      : irn(irn), instanceGraph(instanceGraph) {}

  /// Entrypoint.
  LogicalResult visit(FModuleLike mod);

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

  // Collect force and release operations for later synthesis.
  LogicalResult visitStmt(RefForceOp op);
  LogicalResult visitStmt(RefReleaseOp op);

  // The `_initial` forms are unsupported.
  LogicalResult visitStmt(RefForceInitialOp op) {
    return op.emitError("force_initial not supported");
  }
  LogicalResult visitStmt(RefReleaseInitialOp op) {
    return op.emitError("release_initial not supported");
  }

private:
  /// Map from probe-typed Value's to their non-probe equivalent.
  DenseMap<Value, Value> probeToHWMap;

  /// Exported probe port, recorded before its argument type is rewritten.
  struct ExportedTarget {
    Value probeSrc;
    Value hwSrc;
    Operation *define;
  };
  DenseMap<Value, ExportedTarget> exportedTargets;

  /// Probe paths through which force control cannot be routed.
  DenseMap<Value, Operation *> unsupportedForceDests;

  /// Forceable operations to demote.
  SmallVector<Forceable> forceables;

  /// Operations to delete.
  SmallVector<Operation *> toDelete;

  /// Inner-ref namespace for resolving inner refs.
  hw::InnerRefNamespace &irn;

  /// Keep instance-graph records synchronized with cloned instances.
  InstanceGraph &instanceGraph;

  /// Per-target force state, keyed by hardware value.
  struct TargetState {
    SmallVector<ForceReleaseAccess> accesses;
    Value instanceCtrl;
    SmallVector<Value, 1> inboundCtrls;
  };

  /// First-touch order makes emission deterministic.
  MapVector<Value, TargetState> targets;

  /// Reuse the first materialized value for each `ref.rwprobe` target.
  DenseMap<hw::InnerRefAttr, Value> rwProbeTargetCache;

  void recordRWProbeTarget(hw::InnerRefAttr target, Value data) {
    rwProbeTargetCache.try_emplace(target, data);
  }

  FailureOr<Value> resolveForceDest(Operation *access, Value dest);

  /// Reduce accesses at module-body end so all operands dominate.
  ForceCtrl reduceAccesses(ImplicitLocOpBuilder &builder,
                           ArrayRef<ForceReleaseAccess> accesses);

  LogicalResult
  collectExportedTargets(FModuleLike mod, Block *block,
                         ArrayRef<std::pair<unsigned, Value>> rwProbePorts);

  LogicalResult materializeForceControl(FModuleLike mod);

  LogicalResult buildStateMachineRegisters(Value data, const ForceCtrl &in);

  /// Override reads while leaving the target single-driven. Only ground types
  /// are supported.
  LogicalResult injectReadSideOverride(Value data, Value effForced,
                                       Value effValue);
};

} // end namespace

//===----------------------------------------------------------------------===//
// Visitor: FModuleLike
//===----------------------------------------------------------------------===//

static Block *getBodyBlock(FModuleLike mod) {
  assert(mod->getNumRegions() == 1);
  auto &blocks = mod->getRegion(0).getBlocks();
  return !blocks.empty() ? &blocks.front() : nullptr;
}

static void attachForceDestBlockerNote(InFlightDiagnostic &diag,
                                       Operation *blocker) {
  if (isa<FInstanceLike>(blocker))
    diag.attachNote(blocker->getLoc())
        << "target is a probe of this instance, whose module has no body to "
           "carry the force control";
  else
    diag.attachNote(blocker->getLoc()) << "target is reached through this op";
}

/// Visit a module, converting its ports and internals to use hardware signals
/// instead of probes.
LogicalResult ProbeVisitor::visit(FModuleLike mod) {
  // Create stand-ins for probe ports while rewriting the body.
  SmallVector<std::pair<size_t, WireOp>> wires;

  auto portTypes = mod.getPortTypes();
  auto portLocs = mod.getPortLocationsAttr().getAsRange<Location>();
  auto portNames = mod.getPortNamesAttr();
  SmallVector<Attribute> newPortTypes;

  wires.reserve(portTypes.size());
  newPortTypes.reserve(portTypes.size());
  auto *block = getBodyBlock(mod);
  bool portsToChange = false;
  SmallVector<std::pair<unsigned, FIRRTLBaseType>> forceablePorts;
  for (auto [idx, typeAttr, loc] : llvm::enumerate(portTypes, portLocs)) {
    auto type = cast<TypeAttr>(typeAttr);
    auto conv = convertType(type.getValue(), loc);
    if (failed(conv))
      return failure();
    auto newType = *conv;

    if (!newType) {
      newPortTypes.push_back(type);
      continue;
    }

    portsToChange = true;
    newPortTypes.push_back(TypeAttr::get(newType));

    if (cast<RefType>(type.getValue()).getForceable())
      forceablePorts.emplace_back(idx, type_cast<FIRRTLBaseType>(newType));

    if (!block)
      continue;

    // Stand-in until the signature is updated; RAUW'd to the argument after.
    auto builder = ImplicitLocOpBuilder::atBlockBegin(loc, block);
    auto wire = WireOp::create(builder, newType);
    wires.emplace_back(idx, wire);

    probeToHWMap[block->getArgument(idx)] = wire.getData();
  }

  // Update body, if present.
  if (block &&
      block
          ->walk<mlir::WalkOrder::PreOrder>(
              [&](Operation *op) -> WalkResult { return dispatchVisitor(op); })
          .wasInterrupted())
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

  // Append control inputs without changing existing port indices.
  SmallVector<std::pair<unsigned, Value>> rwProbePorts;
  if (!forceablePorts.empty()) {
    auto *ctx = mod->getContext();
    unsigned appendAt = mod.getNumPorts();

    llvm::StringSet<> taken;
    for (auto name : portNames.getAsRange<StringAttr>())
      taken.insert(name.getValue());

    SmallVector<std::pair<unsigned, PortInfo>> ctrlPorts;
    ctrlPorts.reserve(forceablePorts.size());
    for (auto [idx, probedType] : forceablePorts) {
      SmallString<64> name(cast<StringAttr>(portNames[idx]).getValue());
      name += "_force_ctrl";
      auto baseLen = name.size();
      for (unsigned suffix = 0; !taken.insert(name).second; ++suffix) {
        name.truncate(baseLen);
        (Twine("_") + Twine(suffix)).toVector(name);
      }
      ctrlPorts.emplace_back(
          appendAt,
          PortInfo(StringAttr::get(ctx, name),
                   createForceCtrlBundleType(probedType), Direction::In,
                   /*symName=*/StringAttr{}, mod.getPortLocation(idx)));
    }
    mod.insertPorts(ctrlPorts);

    if (block)
      for (auto [k, port] : llvm::enumerate(forceablePorts))
        rwProbePorts.emplace_back(port.first, block->getArgument(appendAt + k));
  }

  if (block && !rwProbePorts.empty()) {
    if (failed(collectExportedTargets(mod, block, rwProbePorts)))
      return failure();
  }

  if (failed(materializeForceControl(mod)))
    return failure();

  // Delete operations that were converted.
  for (auto *op : llvm::reverse(toDelete))
    op->erase();

  // The synthesized state machine replaces forceability.
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

  // The tapped memory must be a local combinational memory.
  auto mem = op.getMemory().getDefiningOp<chirrtl::CombMemOp>();
  assert(mem);

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

  // Add one read port per address.
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

  assert(vectype == FVectorType::get(mem.getType().getElementType(),
                                     mem.getType().getNumElements()));
  auto vecData = VectorCreateOp::create(builder, vectype, data);

  // Keep existing users dominated by both the replacement and the new reads.
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
  // FIRRTL memory debug ports are not supported here.
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

  // Clone the wire with its converted type.
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto cloned = cast<WireOp>(builder.clone(*op));
  cloned->getOpResults().front().setType(type);
  probeToHWMap[op.getDataRaw()] = cloned.getData();
  toDelete.push_back(op);
  return success();
}

CtrlGroup readCtrlGroup(ImplicitLocOpBuilder &builder, Value bundle) {
  CtrlGroup group;
  group.forceActive =
      getBundleField(builder, bundle, ProbeVisitor::forceActiveName);
  group.releaseActive =
      getBundleField(builder, bundle, ProbeVisitor::releaseActiveName);
  group.forcedValue =
      getBundleField(builder, bundle, ProbeVisitor::forcedValueName);
  return group;
}

static Value readForceCtrlClock(ImplicitLocOpBuilder &builder, Value bundle) {
  return getBundleField(builder, bundle, ProbeVisitor::clockName);
}

/// Fill absent control fields; materialize `forcedValue` when the sink needs a
/// driven value.
CtrlGroup materializeCtrlGroup(ImplicitLocOpBuilder &builder,
                               FIRRTLBaseType probedType, CtrlGroup group,
                               bool tieOffValue) {
  if (!group.forceActive)
    group.forceActive = getU1Const(builder, false);
  if (!group.releaseActive)
    group.releaseActive = getU1Const(builder, false);
  if (!group.forcedValue && tieOffValue)
    group.forcedValue = builder.createOrFold<InvalidValueOp>(probedType);
  return group;
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

  // Reuse the declaration's symbol so aliased RWProbes share the target.
  if (auto sym = hw::InnerSymbolTable::getInnerSymbol(fop)) {
    auto module = fop->getParentOfType<FModuleOp>();
    assert(module && "forceable declaration must be inside an FModuleOp");
    recordRWProbeTarget(
        hw::InnerRefAttr::get(SymbolTable::getSymbolName(module), sym), data);
  }

  probeToHWMap[fop.getDataRef()] = data;
  return success();
}

//===----------------------------------------------------------------------===//
// Read-side override injection
//===----------------------------------------------------------------------===//

static bool isWriteUse(OpOperand &use) {
  if (auto conn = dyn_cast<FConnectLike>(use.getOwner())) {
    // Operand index, not value: `connect a, a` writes dest and reads src.
    assert(conn.getDest() == conn->getOperand(0) && "unexpected connect shape");
    return use.getOperandNumber() == 0;
  }
  return false;
}

static bool redirectReads(Value raw, Value observed,
                          SmallPtrSetImpl<Operation *> &skip) {
  bool redirected = false;
  for (OpOperand &use : llvm::make_early_inc_range(raw.getUses())) {
    Operation *owner = use.getOwner();
    if (skip.contains(owner) || isWriteUse(use))
      continue;
    use.set(observed);
    redirected = true;
  }
  return redirected;
}

static void collectFanInCone(ArrayRef<Value> roots,
                             SmallPtrSetImpl<Operation *> &cone) {
  SmallVector<Value> worklist(roots);
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val().getDefiningOp();
    if (!op || !cone.insert(op).second)
      continue;
    llvm::append_range(worklist, op->getOperands());
  }
}

LogicalResult ProbeVisitor::injectReadSideOverride(Value data, Value effForced,
                                                   Value effValue) {
  auto type = type_dyn_cast<FIRRTLBaseType>(data.getType());
  if (!type || !type.isGround())
    return mlir::emitError(data.getLoc())
           << "force/release of aggregate types is not supported; compile with "
              "preserve-aggregate=none";

  // Keep the control cone unchanged to avoid a mux cycle.
  SmallPtrSet<Operation *, 16> skip;
  collectFanInCone({effForced, effValue}, skip);

  auto *body = getParentModule(data).getBodyBlock();
  Location loc = data.getLoc();

  // Place the observed wire next to the target; control is emitted at block
  // end.
  ImplicitLocOpBuilder wireBuilder(loc, data.getContext());
  if (auto *dataDef = data.getDefiningOp())
    wireBuilder.setInsertionPointAfter(dataDef);
  else
    wireBuilder.setInsertionPointToStart(body);
  SmallString<32> wireName;
  if (auto [name, valid] = getFieldName(FieldRef(data, 0), /*nameSafe=*/true);
      valid)
    wireName = name;
  wireName += "_forced";
  auto observedWire = WireOp::create(wireBuilder, data.getType(), wireName);
  skip.insert(observedWire);
  Value observed = observedWire.getData();

  if (!redirectReads(data, observed, skip)) {
    observedWire.erase();
    return success();
  }

  ImplicitLocOpBuilder builder(loc, body, body->end());
  auto mux = MuxPrimOp::create(builder, effForced, effValue, data);
  skip.insert(mux);
  Value selected = mux.getResult();
  auto connect = MatchingConnectOp::create(builder, observed, selected);
  skip.insert(connect);

  return success();
}

LogicalResult ProbeVisitor::buildStateMachineRegisters(Value data,
                                                       const ForceCtrl &in) {
  Location loc = data.getLoc();
  ImplicitLocOpBuilder builder(loc, data.getContext());

  auto fModule = getParentModule(data);
  assert(fModule && "Expected to find parent FModuleOp");

  if (auto port = dyn_cast<BlockArgument>(data))
    if (fModule.getPortDirection(port.getArgNumber()) == Direction::Out)
      return mlir::emitError(
          loc, "cannot synthesize force/release: target is a module output "
               "port");

  // A source-flow value cannot be driven by the synthesized state.
  if (foldFlow(data) == Flow::Source)
    return mlir::emitError(
        loc, "cannot synthesize force/release: target is read-only "
             "(source flow) and cannot be driven");

  // Nothing observes the target, so there is nothing to override; do not emit
  // dead state or control logic. This check must precede all synthesized ops.
  if (llvm::none_of(data.getUses(),
                    [](OpOperand &use) { return !isWriteUse(use); }))
    return success();

  auto u1Type = UIntType::get(data.getContext(), 1);

  // Sample the event but keep the winning force's RHS live.
  auto *body = fModule.getBodyBlock();
  builder.setInsertionPointToEnd(body);

  CtrlGroup group =
      materializeCtrlGroup(builder, type_cast<FIRRTLBaseType>(data.getType()),
                           in.group, /*tieOffValue=*/true);
  assert(group.forceActive && group.releaseActive && group.forcedValue &&
         "state machine control must be fully materialized");

  auto forcedRegOp = RegOp::create(builder, u1Type, in.clk, "forced");
  forcedRegOp.setInitialAttr(getIntZerosAttr(u1Type));
  Value forcedReg = forcedRegOp.getResult();

  Value cZero = getU1Const(builder, false);
  Value cOne = getU1Const(builder, true);

  MatchingConnectOp::create(builder, forcedReg,
                            builder.createOrFold<MuxPrimOp>(
                                group.forceActive, cOne,
                                builder.createOrFold<MuxPrimOp>(
                                    group.releaseActive, cZero, forcedReg)));

  return injectReadSideOverride(data, forcedReg, group.forcedValue);
}

LogicalResult ProbeVisitor::visitInstanceLike(FInstanceLike oldInst) {
  SmallVector<Type> newTypes;
  auto needsConv =
      mapRange(oldInst->getResultTypes(), oldInst->getLoc(), newTypes);
  if (failed(needsConv))
    return failure();
  if (!*needsConv)
    return success();

  // All referenced modules have the same signature.
  auto aMod = irn.symTable.lookup<FModuleLike>(
      *oldInst.getReferencedModuleNames().begin());
  assert(aMod && "instance must reference an existing module");

  unsigned origNumPorts = oldInst->getNumResults();
  assert(aMod.getNumPorts() >= origNumPorts &&
         "instance results must match the referenced module's ports");

  SmallVector<std::pair<unsigned, PortInfo>> ctrlPorts;
  for (unsigned idx = origNumPorts, e = aMod.getNumPorts(); idx != e; ++idx)
    ctrlPorts.emplace_back(
        idx, PortInfo(aMod.getPortNameAttr(idx), aMod.getPortType(idx),
                      aMod.getPortDirection(idx),
                      /*symName=*/StringAttr{}, aMod.getPortLocation(idx)));

  // Clone the instance with converted results and the callee's control ports.
  auto newInst = oldInst.cloneWithInsertedPorts(ctrlPorts);
  instanceGraph.replaceInstance(oldInst, newInst);

  unsigned ctrlIdx = origNumPorts;
  for (auto [idx, newType] : llvm::enumerate(newTypes)) {
    auto oldResult = oldInst->getOpResult(idx);
    auto newResult = newInst->getOpResult(idx);
    if (newType == oldResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }

    newResult.setType(newType);

    auto refType = cast<RefType>(oldResult.getType());
    probeToHWMap[oldResult] = newResult;
    if (!refType.getForceable())
      continue;

    // Match forceable results with the callee's appended control ports.
    assert(ctrlIdx < aMod.getNumPorts() &&
           aMod.getPortDirection(ctrlIdx) == Direction::In &&
           aMod.getPortType(ctrlIdx) ==
               createForceCtrlBundleType(type_cast<FIRRTLBaseType>(newType)) &&
           "control port out of sync with forceable result");

    // Keep each forceable result as an independent control channel.
    targets[newResult].instanceCtrl = newInst->getOpResult(ctrlIdx++);
  }
  assert(ctrlIdx == aMod.getNumPorts() && "unconsumed control ports");

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

  // Record exports before rewriting the port argument type.
  if (isa<BlockArgument>(op.getDest()))
    exportedTargets[op.getDest()] = {op.getSrc(), newSrc, op};

  // The source must dominate the destination for an equivalent connect.
  assert(!isa<BlockArgument>(newDest));
  auto *destDefiningOp = newDest.getDefiningOp();
  assert(destDefiningOp);
  if (!newSrc.getParentBlock()->findAncestorOpInBlock(*destDefiningOp)) {
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
  // Resolve the target and materialize the selected field.
  auto conv = mapType(op.getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;
  toDelete.push_back(op);

  auto ist = irn.lookup(op.getTarget());
  assert(ist);
  auto ref = getFieldRefForTarget(ist);

  // Reuse one hardware value when multiple RWProbes name the same target.
  if (Value cached = rwProbeTargetCache.lookup(op.getTarget())) {
    probeToHWMap[op.getResult()] = cached;
  } else {
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
    recordRWProbeTarget(op.getTarget(), data);
    probeToHWMap[op.getResult()] = data;
  }

  // Force control covers the whole declaration, not a field-level target.
  if (ref.getFieldID() != 0)
    unsupportedForceDests[op.getResult()] = op;

  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefCastOp op) {
  auto input = probeToHWMap.at(op.getInput());

  auto conv = mapType(op.getResult().getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;
  toDelete.push_back(op);

  // Preserve identity mappings so force control remains keyed by the target.
  if (newType == input.getType()) {
    probeToHWMap[op.getResult()] = input;
    if (auto *blocker = unsupportedForceDests.lookup(op.getInput()))
      unsupportedForceDests[op.getResult()] = blocker;
    return success();
  }

  // A type-changing cast requires a converted copy.
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(input);
  auto wire = WireOp::create(builder, newType);
  emitConnect(builder, wire.getData(), input);
  probeToHWMap[op.getResult()] = wire.getData();

  // A copy wire cannot carry force control.
  if (cast<RefType>(op.getResult().getType()).getForceable())
    unsupportedForceDests[op.getResult()] = op;
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefSendOp op) {
  auto conv = mapType(op.getResult().getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;
  toDelete.push_back(op);

  // Reuse the input when no type conversion is needed.
  if (newType == op.getBase().getType()) {
    probeToHWMap[op.getResult()] = op.getBase();
    return success();
  }

  // Otherwise, create a passive copy.
  assert(newType == op.getBase().getType().getPassiveType());
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(op.getBase());
  auto wire = WireOp::create(builder, newType);
  emitConnect(builder, wire.getData(), op.getBase());
  probeToHWMap[op.getResult()] = wire.getData();
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefResolveOp op) {
  // Replace `ref.resolve` with the mapped value.
  auto val = probeToHWMap.at(op.getRef());
  op.replaceAllUsesWith(val);
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefSubOp op) {
  // Replace `ref.sub` with field selection on the mapped value.
  auto val = probeToHWMap.at(op.getInput());
  assert(val);
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(op.getInput());
  auto newVal =
      getValueByFieldID(builder, val, op.getAccessedField().getFieldID());
  probeToHWMap[op.getResult()] = newVal;
  toDelete.push_back(op);

  // Force control covers the whole target, so field forces are diagnosed.
  if (cast<RefType>(op.getResult().getType()).getForceable()) {
    if (auto *blocker = unsupportedForceDests.lookup(op.getInput()))
      unsupportedForceDests[op.getResult()] = blocker;
    else
      unsupportedForceDests[op.getResult()] = op;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Force/Release Synthesis
//===----------------------------------------------------------------------===//

/// Latch which force is active while keeping its RHS live.
static Value stickyLiveForceValue(ImplicitLocOpBuilder &builder,
                                  ArrayRef<std::pair<Value, Value>> forces,
                                  Value forceActive, Value clk,
                                  StringRef regName) {
  assert(!forces.empty() && "sticky value of a group that never forces");
  if (forces.size() == 1)
    return forces.front().second;

  // Priority has already made the force predicates mutually exclusive.
  auto later = forces.drop_front();

  auto u1Type = UIntType::get(builder.getContext(), 1);
  Value value = forces.front().second;
  SmallVector<Value> wins;
  wins.reserve(later.size());
  for (const auto &force : later) {
    auto winRegOp = RegOp::create(builder, u1Type, clk, regName);
    winRegOp.setInitialAttr(getIntZerosAttr(u1Type));
    wins.push_back(winRegOp.getResult());
    value = builder.createOrFold<MuxPrimOp>(winRegOp.getResult(), force.second,
                                            value);
  }

  for (auto [win, force] : llvm::zip_equal(wins, later))
    MatchingConnectOp::create(
        builder, win,
        builder.createOrFold<MuxPrimOp>(forceActive, force.first, win));

  return value;
}

/// OR-reduce a non-empty list.
static Value orReduce(ImplicitLocOpBuilder &builder, ArrayRef<Value> values) {
  Value result = values.front();
  for (Value value : values.drop_front())
    result = builder.createOrFold<OrPrimOp>(result, value);
  return result;
}

/// Merge control sources so the highest-priority event wins as a whole.
static CtrlGroup reduceCtrlSources(ImplicitLocOpBuilder &builder,
                                   MutableArrayRef<CtrlGroup> sources,
                                   Value clk) {
  // Mask each source when a higher-priority source is active.
  Value higherActive;
  for (size_t idx = sources.size(); idx-- > 0;) {
    CtrlGroup &source = sources[idx];
    // The lowest-priority source is not masked.
    Value active = idx == 0 ? Value()
                            : builder.createOrFold<OrPrimOp>(
                                  source.forceActive, source.releaseActive);
    if (higherActive) {
      Value selected = builder.createOrFold<NotPrimOp>(higherActive);
      source.forceActive =
          builder.createOrFold<AndPrimOp>(source.forceActive, selected);
      source.releaseActive =
          builder.createOrFold<AndPrimOp>(source.releaseActive, selected);
    }
    if (active)
      higherActive = higherActive
                         ? builder.createOrFold<OrPrimOp>(higherActive, active)
                         : active;
  }

  SmallVector<Value> forceActives, releaseActives;
  SmallVector<std::pair<Value, Value>> forces;
  for (const CtrlGroup &source : sources) {
    forceActives.push_back(source.forceActive);
    releaseActives.push_back(source.releaseActive);
    if (source.forcedValue)
      forces.emplace_back(source.forceActive, source.forcedValue);
  }

  Value forceActive = orReduce(builder, forceActives);
  Value releaseActive = orReduce(builder, releaseActives);
  Value forcedValue = forces.empty()
                          ? Value()
                          : stickyLiveForceValue(builder, forces, forceActive,
                                                 clk, "forceWinner");

  return {forceActive, releaseActive, forcedValue};
}

/// Merge inbound controls in port order, then apply local control at highest
/// priority.
CtrlGroup combineCtrlSources(ImplicitLocOpBuilder &builder,
                             FIRRTLBaseType probedType,
                             ArrayRef<CtrlGroup> inbound, CtrlGroup local,
                             Value clk) {
  SmallVector<CtrlGroup> sources(inbound.begin(), inbound.end());
  sources.push_back(
      materializeCtrlGroup(builder, probedType, local, /*tieOffValue=*/false));
  return reduceCtrlSources(builder, sources, clk);
}

/// Drive an instance control bundle, tying absent control off.
void connectControlFields(ImplicitLocOpBuilder &builder, Value control,
                          FIRRTLBaseType probedType, CtrlGroup group,
                          Value clk) {
  // An unforced target still needs a valid clock input.
  if (!clk)
    clk =
        SpecialConstantOp::create(builder, ClockType::get(builder.getContext()),
                                  builder.getBoolAttr(false));
  group = materializeCtrlGroup(builder, probedType, group,
                               /*tieOffValue=*/true);
  auto dst = readCtrlGroup(builder, control);
  Value clkField = readForceCtrlClock(builder, control);
  MatchingConnectOp::create(builder, dst.forceActive, group.forceActive);
  MatchingConnectOp::create(builder, dst.releaseActive, group.releaseActive);
  MatchingConnectOp::create(builder, dst.forcedValue, group.forcedValue);
  MatchingConnectOp::create(builder, clkField, clk);
}

ForceCtrl ProbeVisitor::reduceAccesses(ImplicitLocOpBuilder &builder,
                                       ArrayRef<ForceReleaseAccess> accesses) {
  Value clk = accesses.front().clock;

  // Later accesses have higher priority.
  Value cZero = getU1Const(builder, false);
  SmallVector<CtrlGroup> sources;
  sources.reserve(accesses.size());
  for (auto &access : accesses) {
    if (access.isForce())
      sources.push_back({access.predicate, cZero, access.forceValue.value()});
    else
      sources.push_back({cZero, access.predicate, Value()});
  }

  return {reduceCtrlSources(builder, sources, clk), clk};
}

LogicalResult ProbeVisitor::collectExportedTargets(
    FModuleLike mod, Block *block,
    ArrayRef<std::pair<unsigned, Value>> rwProbePorts) {
  for (auto [portIdx, inbound] : rwProbePorts) {
    auto exportIt = exportedTargets.find(block->getArgument(portIdx));

    if (exportIt == exportedTargets.end())
      return mod->emitError(
                 "forceable probe port cannot be lowered: no ref.define "
                 "exporting a local target for port ")
             << mod.getPortNameAttr(portIdx).getValue();

    const auto &exported = exportIt->second;
    auto refDef = cast<RefDefineOp>(exported.define);
    auto outSrc = exported.probeSrc;

    if (auto *blocker = unsupportedForceDests.lookup(outSrc)) {
      auto diag = refDef.emitError(
          "forceable probe port cannot be lowered: force control cannot be "
          "routed to the target through this probe");
      attachForceDestBlockerNote(diag, blocker);
      return failure();
    }

    // `rwProbePorts` is built in ascending port order, so appending preserves
    // port priority: a later port overrides an earlier port.
    targets[exported.hwSrc].inboundCtrls.push_back(inbound);
  }
  return success();
}

LogicalResult ProbeVisitor::materializeForceControl(FModuleLike mod) {
  auto *block = getBodyBlock(mod);

  for (auto &[hwVal, state] : targets) {
    auto probedType = type_cast<FIRRTLBaseType>(hwVal.getType());

    ForceCtrl local;
    if (!state.accesses.empty()) {
      // The synthesized logic assumes one normalized clock per target.
      const ForceReleaseAccess &first = state.accesses.front();

      ImplicitLocOpBuilder builder(first.op->getLoc(), block, block->end());
      local = reduceAccesses(builder, state.accesses);
    }

    if (!state.inboundCtrls.empty()) {
      ImplicitLocOpBuilder builder(state.inboundCtrls.front().getLoc(), block,
                                   block->end());
      // Preserve port order when merging control bundles.
      SmallVector<CtrlGroup> inboundGroups;
      inboundGroups.reserve(state.inboundCtrls.size());
      for (Value ctrl : state.inboundCtrls)
        inboundGroups.push_back(readCtrlGroup(builder, ctrl));

      Value clk = local.clk;
      if (!clk)
        clk = readForceCtrlClock(builder, state.inboundCtrls.front());

      CtrlGroup group = combineCtrlSources(builder, probedType, inboundGroups,
                                           local.group, clk);

      if (state.instanceCtrl) {
        connectControlFields(builder, state.instanceCtrl, probedType, group,
                             clk);
        continue;
      }

      if (failed(buildStateMachineRegisters(hwVal, ForceCtrl{group, clk})))
        return failure();
      continue;
    }

    if (state.instanceCtrl) {
      auto *ctrlBlock = state.instanceCtrl.getParentBlock();
      ImplicitLocOpBuilder builder(state.instanceCtrl.getLoc(), ctrlBlock,
                                   ctrlBlock->end());
      connectControlFields(builder, state.instanceCtrl, probedType, local.group,
                           local.clk);
      continue;
    }

    assert(!state.accesses.empty());
    if (failed(buildStateMachineRegisters(hwVal, local)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Force/Release operations
//===----------------------------------------------------------------------===//

/// Reject accesses whose nested region cannot be represented at module scope.
static LogicalResult checkForceReleaseNesting(Operation *op, StringRef what) {
  if (op->getParentOfType<LayerBlockOp>())
    return op->emitError() << what << " inside a layerblock is not supported";
  if (op->getParentOfType<WhenOp>() || op->getParentOfType<MatchOp>())
    return op->emitError() << what
                           << " inside a when or match block is not supported";
  return success();
}

FailureOr<Value> ProbeVisitor::resolveForceDest(Operation *access, Value dest) {
  if (auto *blocker = unsupportedForceDests.lookup(dest)) {
    auto diag = access->emitError(
        "unsupported force/release: cannot route force control to the target "
        "through this probe");
    attachForceDestBlockerNote(diag, blocker);
    return failure();
  }
  Value hwDest = probeToHWMap.lookup(dest);
  if (!hwDest)
    return access->emitError(
        "unsupported force/release: unable to determine the target");
  return hwDest;
}

LogicalResult ProbeVisitor::visitStmt(RefForceOp op) {
  if (failed(checkForceReleaseNesting(op, "force")))
    return failure();

  auto hwDest = resolveForceDest(op, op.getDest());
  if (failed(hwDest))
    return failure();
  targets[*hwDest].accesses.push_back(
      {op, op.getPredicate(), op.getSrc(), op.getClock()});
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitStmt(RefReleaseOp op) {
  if (failed(checkForceReleaseNesting(op, "release")))
    return failure();

  auto hwDest = resolveForceDest(op, op.getDest());
  if (failed(hwDest))
    return failure();
  targets[*hwDest].accesses.push_back(
      {op, op.getPredicate(), std::nullopt, op.getClock()});
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

  // Collect clocked roots before changing module signatures.
  SmallVector<Operation *> gatedClockRoots;
  getOperation()->walk([&](Operation *op) {
    auto fop = dyn_cast<Forceable>(op);
    if (isa<RefForceOp, RefReleaseOp>(op) ||
        (fop && isa<RegOp, RegResetOp>(op) && fop.isForceable()))
      gatedClockRoots.push_back(op);
  });

  auto &instanceGraph = getAnalysis<InstanceGraph>();

  // The conversion mutates signatures, so run it sequentially.
  if (!gatedClockRoots.empty()) {
    GatedClockConversion tracer(instanceGraph);
    for (auto *op : gatedClockRoots)
      if (failed(tracer.addRoot(op)))
        return signalPassFailure();
    if (failed(tracer.run()))
      return signalPassFailure();
  }

  hw::InnerRefNamespace irn{getAnalysis<SymbolTable>(),
                            getAnalysis<hw::InnerSymbolTableCollection>()};

  // Convert callees first so callers see final control ports.
  auto result = instanceGraph.walkPostOrder(
      [&](InstanceGraphNode &node) -> LogicalResult {
        auto mod = node.getModule<FModuleLike>();
        ProbeVisitor visitor(irn, instanceGraph);
        return visitor.visit(mod);
      });

  if (failed(result))
    signalPassFailure();
}
