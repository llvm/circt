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
// Force/release on RWProbes is synthesized per target.  Force control from
// outside a module rides inside the converted port type so no ports are
// inserted and the pass stays module-local:
//
//   probe<T>   -> T
//   rwprobe<T> -> { data: T, flip ctrl: { forceActive, releaseActive, ... } }
//
// Only the force/release event is sampled; the winning force's RHS stays live,
// matching Verilog `force a = v`.  The override is injected on reads, not the
// target's driver, so a force overrides the observed value rather than the
// assignment that computes it.
//
// Gated clocks are converted first so synthesized state runs on a free-running
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

constexpr StringRef probePortDataField = "data";
constexpr StringRef probePortCtrlField = "ctrl";

Value getProbePortData(ImplicitLocOpBuilder &builder, Value port) {
  return getBundleField(builder, port, probePortDataField);
}

Value getProbePortCtrl(ImplicitLocOpBuilder &builder, Value port) {
  return getBundleField(builder, port, probePortCtrlField);
}

struct ForceReleaseAccess {
  Operation *op;
  Value predicate;
  std::optional<Value> forceValue;
  Value clock;

  bool isForce() const { return forceValue.has_value(); }
};

/// Nulls mean "no local control"; reduction never produces them.
struct CtrlGroup {
  Value forceActive;
  Value releaseActive;
  Value forcedValue;
};

/// Reduced control plus the clock the synthesized state runs on.
struct ForceCtrl {
  CtrlGroup clocked;
  Value clk;
};

class ProbeVisitor : public FIRRTLVisitor<ProbeVisitor, LogicalResult> {
public:
  ProbeVisitor(hw::InnerRefNamespace &irn, bool carryCtrlInPortType)
      : irn(irn), carryCtrlInPortType(carryCtrlInPortType) {}

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

  /// `{data, flip ctrl}` so inbound force control needs no extra port.
  FailureOr<Type> convertPortType(Type type, Location loc) {
    auto conv = convertType(type, loc);
    if (failed(conv) || !*conv)
      return conv;
    // Only a probe type converts to non-null, so this cast is safe.
    if (!carryCtrlInPortType || !cast<RefType>(type).getForceable())
      return conv;
    return Type(createProbePortType(type_cast<FIRRTLBaseType>(*conv)));
  }

  /// Return the "target" port type, or failure on error.
  FailureOr<Type> mapPortType(Type type, Location loc) {
    auto newType = convertPortType(type, loc);
    if (failed(newType))
      return failure();
    return *newType ? *newType : type;
  }

  /// Map a range of port (or instance result) types, return if changes needed.
  template <typename R>
  FailureOr<bool> mapPortRange(R &&range, Location loc,
                               SmallVectorImpl<Type> &newTypes) {
    newTypes.reserve(llvm::size(range));

    bool anyConverted = false;
    for (auto type : range) {
      auto conv = mapPortType(type, loc);
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
  LogicalResult visitStmt(RefReleaseOp op);

  // The `_initial` flavours are not supported: reject as unsupported.
  LogicalResult visitStmt(RefForceInitialOp op) {
    return op.emitError("force_initial not supported");
  }
  LogicalResult visitStmt(RefReleaseInitialOp op) {
    return op.emitError("release_initial not supported");
  }

private:
  /// Map from probe-typed Value's to their non-probe equivalent.
  DenseMap<Value, Value> probeToHWMap;

  /// Diagnosed instead of applying a force to the wrong target or dropping it.
  DenseMap<Value, Operation *> unsupportedForceDests;

  /// Forceable operations to demote.
  SmallVector<Forceable> forceables;

  /// Operations to delete.
  SmallVector<Operation *> toDelete;

  /// Read-only copy of inner-ref namespace for resolving inner refs.
  hw::InnerRefNamespace &irn;

  /// Keyed by hardware value so aliasing RWProbes share one entry.
  struct TargetState {
    SmallVector<ForceReleaseAccess> accesses;
    Value instanceCtrl;
    Value inboundCtrl;
  };

  /// First-touch order, so emission is deterministic.
  MapVector<Value, TargetState> targets;

  /// Circuit-wide: only force/release designs pay for ctrl-in-port-type.
  bool carryCtrlInPortType;

  FailureOr<Value> resolveForceDest(Operation *access, Value dest);

  /// Emit at end of module body so every access's operands dominate.
  ForceCtrl reduceAccesses(ImplicitLocOpBuilder &builder,
                           FIRRTLBaseType probedType,
                           ArrayRef<ForceReleaseAccess> accesses);

  LogicalResult
  collectExportedTargets(FModuleLike mod, Block *block,
                         ArrayRef<std::pair<unsigned, Value>> rwProbePorts,
                         ArrayRef<Attribute> portNames);

  LogicalResult materializeForceControl(FModuleLike mod);

  LogicalResult buildStateMachineRegisters(FIRRTLBaseType probedType,
                                           Value data, const ForceCtrl &in);

  /// Override reads, not the target's driver, so force is observed immediately
  /// and the target stays single-driven.  Only ground-type targets are
  /// supported; aggregates must be lowered first.
  LogicalResult injectReadSideOverride(Value data, Value effForced,
                                       Value effValue);

  static BundleType createForceCtrlBundleType(FIRRTLBaseType probedType) {
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

  /// `{data, flip ctrl}` so inbound force control needs no extra port.
  static BundleType createProbePortType(FIRRTLBaseType probedType) {
    auto *ctx = probedType.getContext();
    SmallVector<BundleType::BundleElement> elements = {
        {StringAttr::get(ctx, probePortDataField), /*isFlip=*/false,
         probedType},
        {StringAttr::get(ctx, probePortCtrlField), /*isFlip=*/true,
         createForceCtrlBundleType(probedType)},
    };
    return BundleType::get(ctx, elements);
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
  // Ports -> new ports without probe-ness.
  // For all probe ports, insert non-probe duplex values to use
  // as their replacement while rewriting.  Only if has body.
  SmallVector<std::pair<size_t, WireOp>> wires;

  auto portTypes = mod.getPortTypes();
  auto portLocs = mod.getPortLocationsAttr().getAsRange<Location>();
  auto portNames = mod.getPortNamesAttr();
  SmallVector<Attribute> newPortTypes;

  SmallVector<std::pair<unsigned, Value>> rwProbePorts;

  wires.reserve(portTypes.size());
  newPortTypes.reserve(portTypes.size());
  auto *block = getBodyBlock(mod);
  bool portsToChange = false;
  for (auto [idx, typeAttr, loc] : llvm::enumerate(portTypes, portLocs)) {
    auto type = cast<TypeAttr>(typeAttr);
    auto conv = convertPortType(type.getValue(), loc);
    if (failed(conv))
      return failure();
    auto newType = *conv;

    if (!newType) {
      newPortTypes.push_back(type);
      continue;
    }

    portsToChange = true;
    newPortTypes.push_back(TypeAttr::get(newType));
    if (!block)
      continue;

    // Stand-in until the signature is updated; RAUW'd to the argument after.
    auto builder = ImplicitLocOpBuilder::atBlockBegin(loc, block);
    auto wire = WireOp::create(builder, newType);
    wires.emplace_back(idx, wire);

    if (carryCtrlInPortType && cast<RefType>(type.getValue()).getForceable()) {
      probeToHWMap[block->getArgument(idx)] =
          getProbePortData(builder, wire.getData());
      rwProbePorts.emplace_back(idx, getProbePortCtrl(builder, wire.getData()));
    } else
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

  if (block && !rwProbePorts.empty()) {
    if (failed(collectExportedTargets(mod, block, rwProbePorts, portNames)))
      return failure();
  }

  if (failed(materializeForceControl(mod)))
    return failure();

  // Delete operations that were converted.
  for (auto *op : llvm::reverse(toDelete))
    op->erase();

  // Forceability is now the synthesized state machine; drop the rwprobe type.
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

static CtrlGroup readCtrlGroup(ImplicitLocOpBuilder &builder, Value bundle) {
  CtrlGroup group;
  group.forceActive = getBundleField(builder, bundle, "forceActive");
  group.releaseActive = getBundleField(builder, bundle, "releaseActive");
  group.forcedValue = getBundleField(builder, bundle, "forcedValue");
  return group;
}

static Value readForceCtrlClock(ImplicitLocOpBuilder &builder, Value bundle) {
  return getBundleField(builder, bundle, "clk");
}

static ForceCtrl readForceCtrlFields(ImplicitLocOpBuilder &builder,
                                     Value bundle) {
  ForceCtrl fields;
  fields.clocked = readCtrlGroup(builder, bundle);
  fields.clk = readForceCtrlClock(builder, bundle);
  return fields;
}

static Value createU1Const(ImplicitLocOpBuilder &builder, bool value) {
  return builder.createOrFold<ConstantOp>(
      APSInt(APInt(1, value ? 1 : 0, /*isSigned=*/false), /*isUnsigned=*/true));
}

/// Release-only and no-local-control groups both reduce `forceActive` to 0.
static bool isKnownZero(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  return constant && constant.getValue().isZero();
}

/// Fill nulls so "no local control" can share the reduced-group path.
static CtrlGroup materializeCtrlGroup(ImplicitLocOpBuilder &builder,
                                      FIRRTLBaseType probedType,
                                      CtrlGroup group) {
  if (!group.forceActive)
    group.forceActive = createU1Const(builder, false);
  if (!group.releaseActive)
    group.releaseActive = createU1Const(builder, false);
  if (!group.forcedValue)
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

static bool hasRedirectableRead(Value value,
                                const SmallPtrSetImpl<Operation *> &skip) {
  for (OpOperand &use : value.getUses()) {
    if (skip.contains(use.getOwner()) || isWriteUse(use))
      continue;
    return true;
  }
  return false;
}

/// Rewire reads of `raw` to `observed`, leaving the original driver alone.
static void redirectReads(Value raw, Value observed,
                          SmallPtrSetImpl<Operation *> &skip) {

  for (OpOperand &use : llvm::make_early_inc_range(raw.getUses())) {
    Operation *owner = use.getOwner();
    if (skip.contains(owner) || isWriteUse(use))
      continue;
    use.set(observed);
  }
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
              "preserve-aggregates=none";

  // Control cone must keep the raw target or the mux would loop.
  SmallPtrSet<Operation *, 16> skip;
  collectFanInCone({effForced, effValue}, skip);

  if (!hasRedirectableRead(data, skip))
    return success();

  auto *body = getParentModule(data).getBodyBlock();
  Location loc = data.getLoc();

  // Wire next to the target: control is only available at block end.
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

  ImplicitLocOpBuilder builder(loc, body, body->end());
  Operation *lastBefore = body->empty() ? nullptr : &body->back();
  Value selected = builder.createOrFold<MuxPrimOp>(effForced, effValue, data);
  // Keep the generated override operations out of read redirection. The mux
  // may fold away, so only add its defining op when it was actually created.
  if (auto *mux = selected.getDefiningOp();
      mux && !body->empty() && mux != lastBefore && mux == &body->back())
    skip.insert(mux);
  auto connect = MatchingConnectOp::create(builder, observed, selected);
  skip.insert(connect);

  redirectReads(data, observed, skip);
  return success();
}

LogicalResult
ProbeVisitor::buildStateMachineRegisters(FIRRTLBaseType probedType, Value data,
                                         const ForceCtrl &in) {
  Location loc = data.getLoc();
  ImplicitLocOpBuilder builder(loc, data.getContext());

  auto fModule = getParentModule(data);
  assert(fModule && "Expected to find parent FModuleOp");

  auto u1Type = UIntType::get(data.getContext(), 1);

  // Source-flow: the real driver is elsewhere; don't override local reads only.
  if (foldFlow(data) == Flow::Source) {
    mlir::emitError(loc, "cannot synthesize force/release: target is read-only "
                         "(source flow) and cannot be driven");
    return failure();
  }

  assert(in.clocked.forceActive && in.clocked.releaseActive &&
         in.clocked.forcedValue &&
         "state machine control must be fully materialized");

  // Only the force/release event is sampled; the value stays live so the
  // target keeps tracking the winning force's RHS.
  const CtrlGroup &clocked = in.clocked;

  auto *body = fModule.getBodyBlock();
  builder.setInsertionPointToEnd(body);

  auto forcedRegOp = RegOp::create(builder, u1Type, in.clk, "forced");
  forcedRegOp.setInitialAttr(getIntZerosAttr(u1Type));
  Value forcedReg = forcedRegOp.getResult();

  Value cZero = createU1Const(builder, false);
  Value cOne = createU1Const(builder, true);

  builder.create<MatchingConnectOp>(
      forcedReg, builder.createOrFold<MuxPrimOp>(
                     clocked.forceActive, cOne,
                     builder.createOrFold<MuxPrimOp>(clocked.releaseActive,
                                                     cZero, forcedReg)));

  return injectReadSideOverride(data, forcedReg, clocked.forcedValue);
}

LogicalResult ProbeVisitor::visitInstanceLike(FInstanceLike oldInst) {
  SmallVector<Type> newTypes;
  auto needsConv =
      mapPortRange(oldInst->getResultTypes(), oldInst->getLoc(), newTypes);
  if (failed(needsConv))
    return failure();
  if (!*needsConv)
    return success();

  // Body-less callee has no state machine; diagnose force through it later.
  bool bodylessCallee =
      llvm::any_of(oldInst.getReferencedModuleNames(), [&](StringRef name) {
        auto mod = irn.symTable.lookup<FModuleLike>(name);
        return !mod || !getBodyBlock(mod);
      });

  // New instance with converted types.
  // Move users of unconverted results to the new operation.
  ImplicitLocOpBuilder builder(oldInst->getLoc(), oldInst);
  auto *newInst = builder.clone(*oldInst);
  builder.setInsertionPointAfter(newInst);
  for (auto [oldResult, newResult, newType] : llvm::zip_equal(
           oldInst->getOpResults(), newInst->getOpResults(), newTypes)) {
    if (newType == oldResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }

    newResult.setType(newType);

    auto refType = dyn_cast<RefType>(oldResult.getType());
    if (!refType || !refType.getForceable() || !carryCtrlInPortType) {
      probeToHWMap[oldResult] = newResult;
      continue;
    }

    if (bodylessCallee)
      unsupportedForceDests[oldResult] = oldInst;

    Value data = getProbePortData(builder, newResult);
    probeToHWMap[oldResult] = data;

    // Driver of `ctrl` is decided in `materializeForceControl`.
    targets[data].instanceCtrl = getProbePortCtrl(builder, newResult);
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
  return success();
}

LogicalResult ProbeVisitor::visitExpr(RefCastOp op) {
  auto input = probeToHWMap.at(op.getInput());

  auto conv = mapType(op.getResult().getType(), op.getLoc());
  if (failed(conv))
    return failure();
  auto newType = *conv;
  toDelete.push_back(op);

  // Identity mapped type: don't copy, force control is keyed by hardware value.
  if (newType == input.getType()) {
    probeToHWMap[op.getResult()] = input;
    if (auto *blocker = unsupportedForceDests.lookup(op.getInput()))
      unsupportedForceDests[op.getResult()] = blocker;
    return success();
  }

  // Otherwise, insert wire of the new type, and connect to it.

  // y = ref.cast x : probe<t1> -> probe<t2>
  // ->
  // w = firrtl.wire : t2
  // emitConnect(w : t2, map(x): t1)
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  builder.setInsertionPointAfterValue(input);
  auto wire = WireOp::create(builder, newType);
  emitConnect(builder, wire.getData(), input);
  probeToHWMap[op.getResult()] = wire.getData();

  // Copy wire cannot carry force; diagnose force through this cast.
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

  // Force control (local or instance) is whole-target; force through a field
  // is always diagnosed.
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

/// Latch which clocked force is in effect; the RHS stays live so the target
/// tracks it after predicates drop. Last entry wins a tie.
static Value stickyLiveForceValue(ImplicitLocOpBuilder &builder,
                                  ArrayRef<std::pair<Value, Value>> forces,
                                  Value forceActive, Value clk,
                                  StringRef regName) {
  assert(!forces.empty() && "sticky value of a group that never forces");
  if (forces.size() == 1)
    return forces.front().second;

  // Later forces have higher priority; first force is the implicit default.
  auto later = forces.drop_front();
  SmallVector<Value> sel(later.size());
  Value laterActive;
  for (size_t idx = later.size(); idx-- > 0;) {
    Value predicate = later[idx].first;
    sel[idx] = laterActive ? builder.createOrFold<AndPrimOp>(
                                 predicate,
                                 builder.createOrFold<NotPrimOp>(laterActive))
                           : predicate;
    laterActive = laterActive
                      ? builder.createOrFold<OrPrimOp>(laterActive, predicate)
                      : predicate;
  }

  auto u1Type = UIntType::get(builder.getContext(), 1);
  Value value = forces.front().second;
  SmallVector<Value> wins;
  wins.reserve(later.size());
  for (auto [predicate, forceValue] : later) {
    auto winRegOp = RegOp::create(builder, u1Type, clk, regName);
    winRegOp.setInitialAttr(getIntZerosAttr(u1Type));
    wins.push_back(winRegOp.getResult());
    value = builder.createOrFold<MuxPrimOp>(winRegOp.getResult(), forceValue,
                                            value);
  }

  for (auto [win, select] : llvm::zip_equal(wins, sel))
    MatchingConnectOp::create(
        builder, win,
        builder.createOrFold<MuxPrimOp>(forceActive, select, win));

  return value;
}

/// Merge local with inbound; local wins a simultaneous force. Empty local
/// folds to inbound. The winner is latched so the target tracks its live RHS.
static CtrlGroup combineWithInboundCtrl(ImplicitLocOpBuilder &builder,
                                        CtrlGroup local,
                                        FIRRTLBaseType probedType,
                                        CtrlGroup inbound, Value clk) {
  local = materializeCtrlGroup(builder, probedType, local);
  Value iF = inbound.forceActive;
  Value iR = inbound.releaseActive;
  Value iV = inbound.forcedValue;

  Value forceActive = builder.createOrFold<OrPrimOp>(local.forceActive, iF);
  Value releaseActive = builder.createOrFold<OrPrimOp>(local.releaseActive, iR);

  Value forcedValue;
  if (isKnownZero(local.forceActive))
    forcedValue = iV;
  else
    forcedValue = stickyLiveForceValue(
        builder, {{iF, iV}, {local.forceActive, local.forcedValue}},
        forceActive, clk, "forcedByLocal");

  return {forceActive, releaseActive, forcedValue};
}

/// Drive instance `ctrl`. A null group ties it off inactive.
static void connectControlFields(ImplicitLocOpBuilder &builder, Value control,
                                 FIRRTLBaseType probedType, CtrlGroup clocked,
                                 Value clk) {
  // Nothing drives this target from here, so the clock is never observed.
  if (!clk)
    clk =
        SpecialConstantOp::create(builder, ClockType::get(builder.getContext()),
                                  builder.getBoolAttr(false));
  clocked = materializeCtrlGroup(builder, probedType, clocked);
  auto dst = readForceCtrlFields(builder, control);
  builder.createOrFold<MatchingConnectOp>(dst.clocked.forceActive,
                                          clocked.forceActive);
  builder.createOrFold<MatchingConnectOp>(dst.clocked.releaseActive,
                                          clocked.releaseActive);
  builder.createOrFold<MatchingConnectOp>(dst.clocked.forcedValue,
                                          clocked.forcedValue);
  builder.createOrFold<MatchingConnectOp>(dst.clk, clk);
}

ForceCtrl ProbeVisitor::reduceAccesses(ImplicitLocOpBuilder &builder,
                                       FIRRTLBaseType probedType,
                                       ArrayRef<ForceReleaseAccess> accesses) {
  Value clk = accesses.front().clock;

  Value cZero = createU1Const(builder, false);
  Value cOne = createU1Const(builder, true);

  Value forceWins = cZero;
  SmallVector<ForceReleaseAccess> forces, releases;
  for (auto &access : accesses) {
    Value isForceVal = access.isForce() ? cOne : cZero;
    forceWins = builder.createOrFold<MuxPrimOp>(access.predicate, isForceVal,
                                                forceWins);
    (access.isForce() ? forces : releases).push_back(access);
  }

  auto orReduce = [&](ArrayRef<ForceReleaseAccess> set) -> Value {
    if (set.empty())
      return cZero;
    Value v = set.front().predicate;
    for (auto &a : set.drop_front())
      v = builder.createOrFold<OrPrimOp>(v, a.predicate);
    return v;
  };

  Value forceActive = forceWins;

  Value releaseActive =
      releases.empty()
          ? Value(cZero)
          : builder.createOrFold<AndPrimOp>(
                orReduce(releases), builder.createOrFold<NotPrimOp>(forceWins));

  Value forcedValue;
  if (forces.empty()) {
    forcedValue = builder.createOrFold<InvalidValueOp>(probedType);
  } else {
    SmallVector<std::pair<Value, Value>> forcePairs;
    forcePairs.reserve(forces.size());
    for (auto &access : forces)
      forcePairs.emplace_back(access.predicate, access.forceValue.value());
    forcedValue = stickyLiveForceValue(builder, forcePairs, forceActive, clk,
                                       "forceWinner");
  }

  return {{forceActive, releaseActive, forcedValue}, clk};
}

LogicalResult ProbeVisitor::collectExportedTargets(
    FModuleLike mod, Block *block,
    ArrayRef<std::pair<unsigned, Value>> rwProbePorts,
    ArrayRef<Attribute> portNames) {
  for (auto [portIdx, inbound] : rwProbePorts) {
    auto rwProbe = block->getArgument(portIdx);
    // Find the ref.define that exports the local target out of this port.
    RefDefineOp refDef;
    for (auto *o : rwProbe.getUsers())
      if (auto rd = dyn_cast<RefDefineOp>(o)) {
        refDef = rd;
        break;
      }

    if (!refDef)
      return mod->emitError(
                 "forceable probe port cannot be lowered: no ref.define "
                 "exporting a local target for port ")
             << cast<StringAttr>(portNames[portIdx]).getValue();

    auto outSrc = refDef.getSrc();

    if (auto *blocker = unsupportedForceDests.lookup(outSrc)) {
      auto diag = refDef.emitError(
          "forceable probe port cannot be lowered: force control cannot be "
          "routed to the target through this probe");
      attachForceDestBlockerNote(diag, blocker);
      return failure();
    }

    auto hwSrcIt = probeToHWMap.find(outSrc);
    if (hwSrcIt == probeToHWMap.end())
      return refDef.emitError("forceable probe port cannot be lowered: "
                              "exported target has no hardware value");
    Value hwSrc = hwSrcIt->second;

    targets[hwSrc].inboundCtrl = inbound;
  }
  return success();
}

LogicalResult ProbeVisitor::materializeForceControl(FModuleLike mod) {
  auto *block = getBodyBlock(mod);

  for (auto &[hwVal, state] : targets) {
    auto probedType = type_cast<FIRRTLBaseType>(hwVal.getType());

    ForceCtrl local;
    if (!state.accesses.empty()) {
      // One clock per target; gated clocks are already normalized.
      const ForceReleaseAccess &first = state.accesses.front();

      ImplicitLocOpBuilder builder(first.op->getLoc(), mod);
      builder.setInsertionPointToEnd(block);
      local = reduceAccesses(builder, probedType, state.accesses);
    }

    if (state.inboundCtrl) {
      ImplicitLocOpBuilder builder(state.inboundCtrl.getLoc(), block,
                                   block->end());
      CtrlGroup clocked = combineWithInboundCtrl(
          builder, local.clocked, probedType,
          readCtrlGroup(builder, state.inboundCtrl), local.clk);
      Value clk = local.clk ? local.clk
                            : readForceCtrlClock(builder, state.inboundCtrl);

      if (state.instanceCtrl) {
        connectControlFields(builder, state.instanceCtrl, probedType, clocked,
                             clk);
        continue;
      }

      if (failed(buildStateMachineRegisters(probedType, hwVal,
                                            ForceCtrl{clocked, clk})))
        return failure();
      continue;
    }

    if (state.instanceCtrl) {
      auto *ctrlBlock = state.instanceCtrl.getParentBlock();
      ImplicitLocOpBuilder builder(state.instanceCtrl.getLoc(), ctrlBlock,
                                   ctrlBlock->end());
      connectControlFields(builder, state.instanceCtrl, probedType,
                           local.clocked, local.clk);
      continue;
    }

    if (!state.accesses.empty() &&
        failed(buildStateMachineRegisters(probedType, hwVal, local)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Visitor: Force/Release operations
//===----------------------------------------------------------------------===//

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
  auto hwDest = resolveForceDest(op, op.getDest());
  if (failed(hwDest))
    return failure();
  targets[*hwDest].accesses.push_back(
      {op, op.getPredicate(), op.getSrc(), op.getClock()});
  toDelete.push_back(op);
  return success();
}

LogicalResult ProbeVisitor::visitStmt(RefReleaseOp op) {
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

  // Collect gated-clock roots and whether any force/release exists (needed
  // before converting forceable ports to carry ctrl).
  SmallVector<Operation *> gatedClockRoots;
  bool anyForceRelease = false;
  getOperation()->walk([&](Operation *op) {
    if (isa<RefForceOp, RefReleaseOp>(op))
      anyForceRelease = true;
    auto fop = dyn_cast<Forceable>(op);
    if (isa<RefForceOp, RefReleaseOp>(op) ||
        (fop && isa<RegOp, RegResetOp>(op) && fop.isForceable()))
      gatedClockRoots.push_back(op);
  });

  // Sequential: tracer mutates signatures globally. Skip if nothing is clocked.
  if (!gatedClockRoots.empty()) {
    GatedClockConversion tracer(getAnalysis<InstanceGraph>());
    for (auto *op : gatedClockRoots)
      if (failed(tracer.addRoot(op)))
        return signalPassFailure();
    if (failed(tracer.run()))
      return signalPassFailure();
  }

  SmallVector<Operation *, 0> ops(getOperation().getOps<FModuleLike>());

  hw::InnerRefNamespace irn{getAnalysis<SymbolTable>(),
                            getAnalysis<hw::InnerSymbolTableCollection>()};

  auto result = failableParallelForEach(&getContext(), ops, [&](Operation *op) {
    ProbeVisitor visitor(irn, anyForceRelease);
    return visitor.visit(cast<FModuleLike>(op));
  });

  if (result.failed())
    signalPassFailure();
}
