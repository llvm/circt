//===- HoistPassthrough.cpp - Hoist basic passthrough ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HoistPassthrough pass.  This pass identifies basic
// drivers of output ports that can be pulled out of modules.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FieldRefCache.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "firrtl-hoist-passthrough"

using namespace circt;
using namespace firrtl;

using RefValue = mlir::TypedValue<RefType>;

namespace {

struct RefDriver;
struct HWDriver;

//===----------------------------------------------------------------------===//
// (Rematerializable)Driver declaration.
//===----------------------------------------------------------------------===//
/// Statically known driver for a Value.
///
/// Driver source expected to be rematerialized provided a mapping.
/// Generally takes form:
/// [source]----(static indexing?)---->DRIVE_OP---->[dest]
///
/// However, only requirement is that the "driver" can be rematerialized
/// across a module/instance boundary in terms of mapping args<-->results.
///
/// Driver can be reconstructed given a mapping in new location.
///
/// "Update":
/// Map:
///   source -> A
///   dest -> B
///
/// [source]---(indexing)--> SSA_DRIVE_OP ---> [dest]
///   + ([s']---> SSA_DRIVE_OP ---> [A])
///  =>
///  RAUW(B, [A]--(clone indexing))
///  (or RAUW(B, [s']--(clone indexing)))
///
/// Update is safe if driver classification is ""equivalent"" for each context
/// on the other side.  For hoisting U-Turns, this is safe in all cases,
/// for sinking n-turns the driver must be map-equivalent at all instantiation
/// sites.
/// Only UTurns are supported presently.
///
/// The goal is to drop the destination port, so after replacing all users
/// on other side of the instantiation, drop the port driver and move
/// all its users to the driver (immediate) source.
/// This may not be safe if the driver source does not dominate all users of the
/// port, in which case either reject (unsafe) or insert a temporary wire to
/// drive instead.
///
/// RAUW'ing may require insertion of conversion ops if types don't match.
//===----------------------------------------------------------------------===//
struct Driver {
  //-- Data -----------------------------------------------------------------//

  /// Connect entirely and definitively driving the destination.
  FConnectLike drivingConnect;
  /// Source of LHS.
  FieldRef source;

  //-- Constructors ---------------------------------------------------------//
  Driver() = default;
  Driver(FConnectLike connect, FieldRef source)
      : drivingConnect(connect), source(source) {
    assert((isa<RefDriver, HWDriver>(*this)));
  }

  //-- Driver methods -------------------------------------------------------//

  // "Virtual" methods, either commonly defined or dispatched appropriately.

  /// Determine direct driver for the given value, empty Driver otherwise.
  static Driver get(Value v, FieldRefCache &refs);

  /// Whether this can be rematerialized up through an instantiation.
  bool canHoist() const { return isa<BlockArgument>(source.getValue()); }

  /// Simple mapping across instantiation by index.
  using PortMappingFn = llvm::function_ref<Value(size_t)>;

  /// Rematerialize this driven value, using provided mapping function and
  /// builder. New value is returned.
  Value remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder);

  /// Drop uses of the destination, inserting temporary as necessary.
  /// Erases the driving connection, invalidating this Driver.
  void finalize(ImplicitLocOpBuilder &builder);

  //--- Helper methods -------------------------------------------------------//

  /// Return whether this driver is valid/non-null.
  operator bool() const { return source; }

  /// Get driven destination value.
  Value getDest() const {
    // (const cast to workaround getDest() not being const, even if mutates the
    // Operation* that's fine)
    return const_cast<Driver *>(this)->drivingConnect.getDest();
  }

  /// Whether this driver destination is a module port.
  bool drivesModuleArg() const {
    auto arg = dyn_cast<BlockArgument>(getDest());
    assert(!arg || isa<firrtl::FModuleLike>(arg.getOwner()->getParentOp()));
    return !!arg;
  }

  /// Whether this driver destination is an instance result.
  bool drivesInstanceResult() const {
    return getDest().getDefiningOp<hw::HWInstanceLike>();
  }

  /// Get destination as block argument.
  BlockArgument getDestBlockArg() const {
    assert(drivesModuleArg());
    return dyn_cast<BlockArgument>(getDest());
  }

  /// Get destination as operation result, must be instance result.
  OpResult getDestOpResult() const {
    assert(drivesInstanceResult());
    return dyn_cast<OpResult>(getDest());
  }

  /// Helper to obtain argument/result number of destination.
  /// Must be block arg or op result.
  static size_t getIndex(Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg.getArgNumber();
    auto result = dyn_cast<OpResult>(v);
    assert(result);
    return result.getResultNumber();
  }
};

/// Driver implementation for probes.
struct RefDriver : public Driver {
  using Driver::Driver;

  static bool classof(const Driver *t) { return isa<RefValue>(t->getDest()); }

  static RefDriver get(Value v, FieldRefCache &refs);

  Value remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder);
};
static_assert(sizeof(RefDriver) == sizeof(Driver),
              "passed by value, no slicing");

// Driver implementation for HW signals.
// Split out because has more complexity re:safety + updating.
// And can't walk through temporaries in same way.
struct HWDriver : public Driver {
  using Driver::Driver;

  static bool classof(const Driver *t) { return !isa<RefValue>(t->getDest()); }

  static HWDriver get(Value v, FieldRefCache &refs);

  Value remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder);
};
static_assert(sizeof(HWDriver) == sizeof(Driver),
              "passed by value, no slicing");

/// Print driver information.
template <typename T>
static inline T &operator<<(T &os, Driver &d) {
  if (!d)
    return os << "(null)";
  return os << d.getDest() << " <-- " << d.drivingConnect << " <-- "
            << d.source.getValue() << "@" << d.source.getFieldID();
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver implementation.
//===----------------------------------------------------------------------===//

Driver Driver::get(Value v, FieldRefCache &refs) {
  if (auto refDriver = RefDriver::get(v, refs))
    return refDriver;
  if (auto hwDriver = HWDriver::get(v, refs))
    return hwDriver;
  return {};
}

Value Driver::remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder) {
  return TypeSwitch<Driver *, Value>(this)
      .Case<RefDriver, HWDriver>(
          [&](auto *d) { return d->remat(mapPortFn, builder); })
      .Default({});
}

void Driver::finalize(ImplicitLocOpBuilder &builder) {
  auto immSource = drivingConnect.getSrc();
  auto dest = getDest();
  assert(immSource.getType() == dest.getType() &&
         "final connect must be strict");
  if (dest.hasOneUse()) {
    // Only use is the connect, just drop it.
    drivingConnect.erase();
  } else if (isa<BlockArgument>(immSource)) {
    // Block argument dominates all, so drop connect and RAUW to it.
    drivingConnect.erase();
    dest.replaceAllUsesWith(immSource);
  } else {
    // Insert wire temporary.
    // For hoisting use-case could also remat using cached indexing inside the
    // module, but wires keep this simple.
    auto temp = builder.create<WireOp>(immSource.getType());
    dest.replaceAllUsesWith(temp.getDataRaw());
  }
}

//===----------------------------------------------------------------------===//
// RefDriver implementation.
//===----------------------------------------------------------------------===//

static RefDefineOp getRefDefine(Value result) {
  for (auto *user : result.getUsers()) {
    if (auto rd = dyn_cast<RefDefineOp>(user); rd && rd.getDest() == result)
      return rd;
  }
  return {};
}

RefDriver RefDriver::get(Value v, FieldRefCache &refs) {
  auto refVal = dyn_cast<RefValue>(v);
  if (!refVal)
    return {};

  auto rd = getRefDefine(v);
  if (!rd)
    return {};

  auto ref = refs.getFieldRefFromValue(rd.getSrc(), true);
  if (!ref)
    return {};

  return RefDriver(rd, ref);
}

Value RefDriver::remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder) {
  auto mappedSource = mapPortFn(getIndex(source.getValue()));
  auto newVal = getValueByFieldID(builder, mappedSource, source.getFieldID());
  auto destType = getDest().getType();
  if (newVal.getType() != destType)
    newVal = builder.create<RefCastOp>(destType, newVal);
  return newVal;
}

//===----------------------------------------------------------------------===//
// HWDriver implementation.
//===----------------------------------------------------------------------===//

static bool hasDontTouchOrInnerSymOnResult(Operation *op) {
  if (AnnotationSet::hasDontTouch(op))
    return true;
  auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(op);
  return symOp && symOp.getTargetResultIndex() && symOp.getInnerSymAttr();
}

static bool hasDontTouchOrInnerSymOnResult(Value value) {
  if (auto *op = value.getDefiningOp())
    return hasDontTouchOrInnerSymOnResult(op);
  auto arg = dyn_cast<BlockArgument>(value);
  auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
  return (module.getPortSymbolAttr(arg.getArgNumber())) ||
         AnnotationSet::forPort(module, arg.getArgNumber()).hasDontTouch();
}

HWDriver HWDriver::get(Value v, FieldRefCache &refs) {
  auto baseValue = dyn_cast<FIRRTLBaseValue>(v);
  if (!baseValue)
    return {};

  // Output must be passive, for flow reasons.
  // Reject aggregates for now, to be conservative re:aliasing writes/etc.
  // before ExpandWhens.
  if (!baseValue.getType().isPassive() || !baseValue.getType().isGround())
    return {};

  auto connect = getSingleConnectUserOf(v);
  if (!connect)
    return {};

  auto ref = refs.getFieldRefFromValue(connect.getSrc());
  if (!ref)
    return {};

  // Reject if not all same block.
  if (v.getParentBlock() != ref.getValue().getParentBlock() ||
      v.getParentBlock() != connect->getBlock())
    return {};

  // Reject if cannot reason through this.
  // Use local "hasDontTouch" to distinguish inner symbols on results
  // vs on the operation itself (like an instance).
  if (hasDontTouchOrInnerSymOnResult(v) ||
      hasDontTouchOrInnerSymOnResult(ref.getValue()))
    return {};
  if (auto fop = ref.getValue().getDefiningOp<Forceable>();
      fop && fop.isForceable())
    return {};

  // Limit to passive sources for now.
  auto sourceType = type_dyn_cast<FIRRTLBaseType>(ref.getValue().getType());
  if (!sourceType)
    return {};
  if (!sourceType.isPassive())
    return {};

  assert(hw::FieldIdImpl::getFinalTypeByFieldID(sourceType, ref.getFieldID()) ==
             baseValue.getType() &&
         "unexpected type mismatch, cast or extension?");

  return HWDriver(connect, ref);
}

Value HWDriver::remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder) {
  auto mappedSource = mapPortFn(getIndex(source.getValue()));
  // TODO: Cast if needed.  For now only support matching.
  // (No cast needed for current HWDriver's, getFieldRefFromValue and
  // assert)
  return getValueByFieldID(builder, mappedSource, source.getFieldID());
}

//===----------------------------------------------------------------------===//
// MustDrivenBy analysis.
//===----------------------------------------------------------------------===//
namespace {
/// Driver analysis, tracking values that "must be driven" by the specified
/// source (+fieldID), along with final complete driving connect.
class MustDrivenBy {
public:
  MustDrivenBy() = default;
  MustDrivenBy(FModuleOp mod) { run(mod); }

  /// Get direct driver, if computed, for the specified value.
  Driver getDriverFor(Value v) const { return driverMap.lookup(v); }

  /// Get combined driver for the specified value.
  /// Walks the driver "graph" from the value to its ultimate source.
  Driver getCombinedDriverFor(Value v) const {
    Driver driver = driverMap.lookup(v);
    if (!driver)
      return driver;

    // Chase and collapse.
    Driver cur = driver;
    size_t len = 1;
    SmallPtrSet<Value, 8> seen;
    while ((cur = driverMap.lookup(cur.source.getValue()))) {
      // If re-encounter same value, bail.
      if (!seen.insert(cur.source.getValue()).second)
        return {};
      driver.source = cur.source.getSubField(driver.source.getFieldID());
      ++len;
    }
    (void)len;
    LLVM_DEBUG(llvm::dbgs() << "Found driver for " << v << " (chain length = "
                            << len << "): " << driver << "\n");
    return driver;
  }

  /// Analyze the given module's ports and chase simple storage.
  void run(FModuleOp mod) {
    SmallVector<Value, 64> worklist(mod.getArguments());

    DenseSet<Value> enqueued;
    enqueued.insert(worklist.begin(), worklist.end());
    FieldRefCache refs;
    while (!worklist.empty()) {
      auto val = worklist.pop_back_val();
      auto driver =
          ignoreHWDrivers ? RefDriver::get(val, refs) : Driver::get(val, refs);
      driverMap.insert({val, driver});
      if (!driver)
        continue;

      auto sourceVal = driver.source.getValue();

      // If already enqueued, ignore.
      if (!enqueued.insert(sourceVal).second)
        continue;

      // Only chase through atomic values for now.
      // Here, atomic implies must be driven entirely.
      // This is true for HW types, and is true for RefType's because
      // while they can be indexed into, only RHS can have indexing.
      if (hw::FieldIdImpl::getMaxFieldID(sourceVal.getType()) != 0)
        continue;

      // Only through Wires, block arguments, instance results.
      if (!isa<BlockArgument>(sourceVal) &&
          !isa_and_nonnull<WireOp, InstanceOp>(sourceVal.getDefiningOp()))
        continue;

      worklist.push_back(sourceVal);
    }

    refs.verify();

    LLVM_DEBUG({
      llvm::dbgs() << "Analyzed " << mod.getModuleName() << " and found "
                   << driverMap.size() << " drivers.\n";
      refs.printStats(llvm::dbgs());
    });
  }

  /// Clear out analysis results and storage.
  void clear() { driverMap.clear(); }

  /// Configure whether HW signals are analyzed.
  void setIgnoreHWDrivers(bool ignore) { ignoreHWDrivers = ignore; }

private:
  /// Map of values to their computed direct must-drive source.
  DenseMap<Value, Driver> driverMap;
  bool ignoreHWDrivers = false;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {

struct HoistPassthroughPass
    : public HoistPassthroughBase<HoistPassthroughPass> {
  using HoistPassthroughBase::HoistPassthroughBase;
  void runOnOperation() override;

  using HoistPassthroughBase::hoistHWDrivers;
};
} // end anonymous namespace

void HoistPassthroughPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===- Running HoistPassthrough Pass "
                             "------------------------------------------===\n");
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  SmallVector<FModuleOp, 0> modules(llvm::make_filter_range(
      llvm::map_range(
          llvm::post_order(&instanceGraph),
          [](auto *node) { return dyn_cast<FModuleOp>(*node->getModule()); }),
      [](auto module) { return module; }));

  MustDrivenBy driverAnalysis;
  driverAnalysis.setIgnoreHWDrivers(!hoistHWDrivers);

  bool anyChanged = false;

  // For each module (PO)...
  for (auto module : modules) {
    // TODO: Public means can't reason down into, or remove ports.
    // Does not mean cannot clone out wires or optimize w.r.t its contents.
    if (module.isPublic())
      continue;

    // 1. Analyze.

    // What ports to delete.
    // Hoisted drivers of output ports will be deleted.
    BitVector deadPorts(module.getNumPorts());

    // Instance graph node, for walking instances of this module.
    auto *igNode = instanceGraph.lookup(module);

    // Analyze all ports using current IR.
    driverAnalysis.clear();
    driverAnalysis.run(module);
    auto notNullAndCanHoist = [](const Driver &d) -> bool {
      return d && d.canHoist();
    };
    SmallVector<Driver, 16> drivers(llvm::make_filter_range(
        llvm::map_range(module.getArguments(),
                        [&driverAnalysis](auto val) {
                          return driverAnalysis.getCombinedDriverFor(val);
                        }),
        notNullAndCanHoist));

    // If no hoistable drivers found, nothing to do.  Onwards!
    if (drivers.empty())
      continue;

    anyChanged = true;

    // 2. Rematerialize must-driven ports at instantiation sites.

    // Do this first, keep alive Driver state pointing to module.
    for (auto &driver : drivers) {
      std::optional<size_t> deadPort;
      {
        auto destArg = driver.getDestBlockArg();
        auto index = destArg.getArgNumber();

        // Replace dest in all instantiations.
        for (auto *record : igNode->uses()) {
          auto inst = cast<InstanceOp>(record->getInstance());
          ImplicitLocOpBuilder builder(inst.getLoc(), inst);
          builder.setInsertionPointAfter(inst);

          auto mappedDest = inst.getResult(index);
          mappedDest.replaceAllUsesWith(driver.remat(
              [&inst](size_t index) { return inst.getResult(index); },
              builder));
        }
        // The driven port has no external users, will soon be dead.
        deadPort = index;
      }
      assert(deadPort.has_value());

      assert(!deadPorts.test(*deadPort));
      deadPorts.set(*deadPort);

      // Update statistics.
      TypeSwitch<Driver *, void>(&driver)
          .Case<RefDriver>([&](auto *) { ++numRefDrivers; })
          .Case<HWDriver>([&](auto *) { ++numHWDrivers; });
    }

    // 3. Finalize stage.  Ensure remat'd dest is unused on original side.

    ImplicitLocOpBuilder builder(module.getLoc(), module.getBody());
    for (auto &driver : drivers) {
      // Finalize.  Invalidates the driver.
      builder.setLoc(driver.getDest().getLoc());
      driver.finalize(builder);
    }

    // 4. Delete newly dead ports.

    // Drop dead ports at instantiation sites.
    for (auto *record : llvm::make_early_inc_range(igNode->uses())) {
      auto inst = cast<InstanceOp>(record->getInstance());
      ImplicitLocOpBuilder builder(inst.getLoc(), inst);

      assert(inst.getNumResults() == deadPorts.size());
      auto newInst = inst.erasePorts(builder, deadPorts);
      instanceGraph.replaceInstance(inst, newInst);
      inst.erase();
    }

    // Drop dead ports from module.
    module.erasePorts(deadPorts);

    numUTurnsHoisted += deadPorts.count();
  }
  markAnalysesPreserved<InstanceGraph>();

  if (!anyChanged)
    markAllAnalysesPreserved();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createHoistPassthroughPass(bool hoistHWDrivers) {
  auto pass = std::make_unique<HoistPassthroughPass>();
  pass->hoistHWDrivers = hoistHWDrivers;
  return pass;
}
