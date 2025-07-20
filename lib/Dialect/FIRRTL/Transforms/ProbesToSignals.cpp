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

class ProbeVisitor : public FIRRTLVisitor<ProbeVisitor, LogicalResult> {
public:
  ProbeVisitor(hw::InnerRefNamespace &irn) : irn(irn) {}

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

  LogicalResult visitInstanceLike(Operation *op);
  LogicalResult visitDecl(InstanceOp op) { return visitInstanceLike(op); }
  LogicalResult visitDecl(InstanceChoiceOp op) { return visitInstanceLike(op); }

  // Probe operations.

  LogicalResult visitExpr(RWProbeOp op);
  LogicalResult visitExpr(RefCastOp op);
  LogicalResult visitExpr(RefResolveOp op);
  LogicalResult visitExpr(RefSendOp op);
  LogicalResult visitExpr(RefSubOp op);

  LogicalResult visitStmt(RefDefineOp op);

  // Force and release operations: reject as unsupported.
  LogicalResult visitStmt(RefForceOp op) {
    return op.emitError("force not supported");
  }
  LogicalResult visitStmt(RefForceInitialOp op) {
    return op.emitError("force_initial not supported");
  }
  LogicalResult visitStmt(RefReleaseOp op) {
    return op.emitError("release not supported");
  }
  LogicalResult visitStmt(RefReleaseInitialOp op) {
    return op.emitError("release_initial not supported");
  }

private:
  /// Map from probe-typed Value's to their non-probe equivalent.
  DenseMap<Value, Value> probeToHWMap;

  /// Forceable operations to demote.
  SmallVector<Forceable> forceables;

  /// Operations to delete.
  SmallVector<Operation *> toDelete;

  /// Read-only copy of inner-ref namespace for resolving inner refs.
  hw::InnerRefNamespace &irn;
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
  // If module has strings describing XMR suffixes for its ports, reject.
  if (auto internalPaths = mod->getAttrOfType<ArrayAttr>("internalPaths"))
    return mod.emitError("cannot convert module with internal path");

  // Ports -> new ports without probe-ness.
  // For all probe ports, insert non-probe duplex values to use
  // as their replacement while rewriting.  Only if has body.
  SmallVector<std::pair<size_t, WireOp>> wires;

  auto portTypes = mod.getPortTypes();
  auto portLocs = mod.getPortLocationsAttr().getAsRange<Location>();
  SmallVector<Attribute> newPortTypes;

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
        wires.emplace_back(idx, builder.create<WireOp>(loc, newType));
        probeToHWMap[block->getArgument(idx)] = wires.back().second.getData();
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

  // Delete operations that were converted.
  for (auto *op : llvm::reverse(toDelete))
    op->erase();

  // Demote forceable's.
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
    auto port = builder.create<chirrtl::MemoryPortOp>(
        mem.getType().getElementType(),
        chirrtl::CMemoryPortType::get(builder.getContext()), mem.getResult(),
        MemDirAttr::Read, builder.getStringAttr("memTap_" + Twine(i)),
        builder.getArrayAttr({}));
    builder.create<chirrtl::MemoryPortAccessOp>(
        port.getPort(),
        builder.create<ConstantOp>(uintType, APSInt::getUnsigned(i)), clock);
    data.push_back(port.getData());
  }

  // Package up all the reads into a vector.
  assert(vectype == FVectorType::get(mem.getType().getElementType(),
                                     mem.getType().getNumElements()));
  auto vecData = builder.create<VectorCreateOp>(vectype, data);

  // While the new ports are added as late as possible, the debug port
  // operation we're replacing likely has users and those are before
  // the new ports.  Add a wire at a point we know dominates this operation
  // and the new port access operations added above.  This will be used for
  // the existing users of the debug port.
  builder.setInsertionPoint(mem);
  auto wire = builder.create<WireOp>(vectype);
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
    auto wire = builder.create<WireOp>(newType);
    emitConnect(builder, wire.getData(), data);
    data = wire.getData();
  }
  probeToHWMap[fop.getDataRef()] = data;
  return success();
}

LogicalResult ProbeVisitor::visitInstanceLike(Operation *op) {
  SmallVector<Type> newTypes;
  auto needsConv = mapRange(op->getResultTypes(), op->getLoc(), newTypes);
  if (failed(needsConv))
    return failure();
  if (!*needsConv)
    return success();

  // New instance with converted types.
  // Move users of unconverted results to the new operation.
  ImplicitLocOpBuilder builder(op->getLoc(), op);
  auto *newInst = builder.clone(*op);
  for (auto [oldResult, newResult, newType] :
       llvm::zip_equal(op->getOpResults(), newInst->getOpResults(), newTypes)) {
    if (newType == oldResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }

    newResult.setType(newType);
    probeToHWMap[oldResult] = newResult;
  }

  toDelete.push_back(op);
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
    auto wire = builder.create<WireOp>(newType);
    emitConnect(builder, wire.getData(), data);
    data = wire.getData();
  }
  probeToHWMap[op.getResult()] = data;
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
  auto wire = builder.create<WireOp>(newType);
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
  auto wire = builder.create<WireOp>(newType);
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
  LLVM_DEBUG(debugPassHeader(this) << "\n");

  SmallVector<Operation *, 0> ops(getOperation().getOps<FModuleLike>());

  hw::InnerRefNamespace irn{getAnalysis<SymbolTable>(),
                            getAnalysis<hw::InnerSymbolTableCollection>()};

  auto result = failableParallelForEach(&getContext(), ops, [&](Operation *op) {
    ProbeVisitor visitor(irn);
    return visitor.visit(cast<FModuleLike>(op));
  });

  if (result.failed())
    signalPassFailure();
}
