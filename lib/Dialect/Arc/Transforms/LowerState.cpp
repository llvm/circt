//===- LowerState.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-state"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERSTATE
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace hw;
using namespace mlir;
using llvm::SmallDenseSet;

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

namespace {

/// Statistics gathered throughout the execution of this pass.
struct Statistics {
  Pass *parent;
  Statistics(Pass *parent) : parent(parent) {}
  using Statistic = Pass::Statistic;

  Statistic matOpsMoved{parent, "mat-ops-moved",
                        "Ops moved during value materialization"};
  Statistic matOpsCloned{parent, "mat-ops-cloned",
                         "Ops cloned during value materialization"};
  Statistic opsPruned{parent, "ops-pruned", "Ops removed as dead code"};
};

/// Lowering info associated with a single primary clock.
struct ClockLowering {
  /// The root clock this lowering is for.
  Value clock;
  /// A `ClockTreeOp` or `PassThroughOp`  or `InitialOp`.
  Operation *treeOp;
  /// Pass statistics.
  Statistics &stats;
  OpBuilder builder;
  /// A mapping from values outside the clock tree to their materialize form
  /// inside the clock tree.
  IRMapping materializedValues;
  /// A cache of AND gates created for aggregating enable conditions.
  DenseMap<std::pair<Value, Value>, Value> andCache;
  /// A cache of OR gates created for aggregating enable conditions.
  DenseMap<std::pair<Value, Value>, Value> orCache;

  ClockLowering(Value clock, Operation *treeOp, Statistics &stats)
      : clock(clock), treeOp(treeOp), stats(stats), builder(treeOp) {
    assert((isa<ClockTreeOp, PassThroughOp, InitialOp>(treeOp)));
    builder.setInsertionPointToStart(&treeOp->getRegion(0).front());
  }

  Value materializeValue(Value value);
  Value getOrCreateAnd(Value lhs, Value rhs, Location loc);
  Value getOrCreateOr(Value lhs, Value rhs, Location loc);
};

struct GatedClockLowering {
  /// Lowering info of the primary clock.
  ClockLowering &clock;
  /// An optional enable condition of the primary clock. May be null.
  Value enable;
};

/// State lowering for a single `HWModuleOp`.
struct ModuleLowering {
  HWModuleOp moduleOp;
  /// Pass statistics.
  Statistics &stats;
  MLIRContext *context;
  DenseMap<Value, std::unique_ptr<ClockLowering>> clockLowerings;
  DenseMap<Value, GatedClockLowering> gatedClockLowerings;
  std::unique_ptr<ClockLowering> initialLowering;
  Value storageArg;
  OpBuilder clockBuilder;
  OpBuilder stateBuilder;

  ModuleLowering(HWModuleOp moduleOp, Statistics &stats)
      : moduleOp(moduleOp), stats(stats), context(moduleOp.getContext()),
        clockBuilder(moduleOp), stateBuilder(moduleOp) {}

  GatedClockLowering getOrCreateClockLowering(Value clock);
  ClockLowering &getOrCreatePassThrough();
  ClockLowering &getOrCreateInitial();
  Value replaceValueWithStateRead(Value value, Value state);

  void addStorageArg();
  LogicalResult lowerPrimaryInputs();
  LogicalResult lowerPrimaryOutputs();
  LogicalResult lowerStates();
  template <typename CallTy>
  LogicalResult lowerStateLike(Operation *op, Value clock, Value enable,
                               Value reset, ArrayRef<Value> inputs,
                               FlatSymbolRefAttr callee);
  LogicalResult lowerState(StateOp stateOp);
  LogicalResult lowerState(sim::DPICallOp dpiCallOp);
  LogicalResult lowerState(MemoryOp memOp);
  LogicalResult lowerState(MemoryWritePortOp memWriteOp);
  LogicalResult lowerState(TapOp tapOp);
  LogicalResult lowerExtModules(SymbolTable &symtbl);
  LogicalResult lowerExtModule(InstanceOp instOp);

  LogicalResult cleanup();
};
} // namespace

//===----------------------------------------------------------------------===//
// Clock Lowering
//===----------------------------------------------------------------------===//

static bool shouldMaterialize(Operation *op) {
  // Don't materialize arc uses with latency >0, since we handle these in a
  // second pass once all other operations have been moved to their respective
  // clock trees.
  return !isa<MemoryOp, AllocStateOp, AllocMemoryOp, AllocStorageOp,
              ClockTreeOp, PassThroughOp, RootInputOp, RootOutputOp,
              StateWriteOp, MemoryWritePortOp, igraph::InstanceOpInterface,
              StateOp, sim::DPICallOp>(op);
}

static bool shouldMaterialize(Value value) {
  assert(value);

  // Block arguments are just used as they are.
  auto *op = value.getDefiningOp();
  if (!op)
    return false;

  return shouldMaterialize(op);
}

/// Materialize a value within this clock tree. This clones or moves all
/// operations required to produce this value inside the clock tree.
Value ClockLowering::materializeValue(Value value) {
  if (!value)
    return {};
  if (auto mapped = materializedValues.lookupOrNull(value))
    return mapped;
  if (!shouldMaterialize(value))
    return value;

  struct WorkItem {
    Operation *op;
    SmallVector<Value, 2> operands;
    WorkItem(Operation *op) : op(op) {}
  };

  SmallPtrSet<Operation *, 8> seen;
  SmallVector<WorkItem> worklist;

  auto addToWorklist = [&](Operation *outerOp) {
    SmallDenseSet<Value> seenOperands;
    auto &workItem = worklist.emplace_back(outerOp);
    outerOp->walk([&](Operation *innerOp) {
      for (auto operand : innerOp->getOperands()) {
        // Skip operands that are defined within the operation itself.
        if (!operand.getParentBlock()->getParentOp()->isProperAncestor(outerOp))
          continue;

        // Skip operands that we have already seen.
        if (!seenOperands.insert(operand).second)
          continue;

        // Skip operands that we have already materialized or that should not
        // be materialized at all.
        if (materializedValues.contains(operand) || !shouldMaterialize(operand))
          continue;

        workItem.operands.push_back(operand);
      }
    });
  };

  seen.insert(value.getDefiningOp());
  addToWorklist(value.getDefiningOp());

  while (!worklist.empty()) {
    auto &workItem = worklist.back();
    if (!workItem.operands.empty()) {
      auto operand = workItem.operands.pop_back_val();
      if (materializedValues.contains(operand) || !shouldMaterialize(operand))
        continue;
      auto *defOp = operand.getDefiningOp();
      if (!seen.insert(defOp).second) {
        defOp->emitError("combinational loop detected");
        return {};
      }
      addToWorklist(defOp);
    } else {
      builder.clone(*workItem.op, materializedValues);
      seen.erase(workItem.op);
      worklist.pop_back();
    }
  }

  return materializedValues.lookup(value);
}

/// Create an AND gate if none with the given operands already exists. Note that
/// the operands may be null, in which case the function will return the
/// non-null operand, or null if both operands are null.
Value ClockLowering::getOrCreateAnd(Value lhs, Value rhs, Location loc) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  auto &slot = andCache[std::make_pair(lhs, rhs)];
  if (!slot)
    slot = builder.create<comb::AndOp>(loc, lhs, rhs);
  return slot;
}

/// Create an OR gate if none with the given operands already exists. Note that
/// the operands may be null, in which case the function will return the
/// non-null operand, or null if both operands are null.
Value ClockLowering::getOrCreateOr(Value lhs, Value rhs, Location loc) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  auto &slot = orCache[std::make_pair(lhs, rhs)];
  if (!slot)
    slot = builder.create<comb::OrOp>(loc, lhs, rhs);
  return slot;
}

//===----------------------------------------------------------------------===//
// Module Lowering
//===----------------------------------------------------------------------===//

GatedClockLowering ModuleLowering::getOrCreateClockLowering(Value clock) {
  // Look through clock gates.
  if (auto ckgOp = clock.getDefiningOp<seq::ClockGateOp>()) {
    // Reuse the existing lowering for this clock gate if possible.
    if (auto it = gatedClockLowerings.find(clock);
        it != gatedClockLowerings.end())
      return it->second;

    // Get the lowering for the parent clock gate's input clock. This will give
    // us the clock tree to emit things into, alongside the compound enable
    // condition of all the clock gates along the way to the primary clock. All
    // we have to do is to add this clock gate's condition to that list.
    auto info = getOrCreateClockLowering(ckgOp.getInput());
    auto ckgEnable = info.clock.materializeValue(ckgOp.getEnable());
    auto ckgTestEnable = info.clock.materializeValue(ckgOp.getTestEnable());
    info.enable = info.clock.getOrCreateAnd(
        info.enable,
        info.clock.getOrCreateOr(ckgEnable, ckgTestEnable, ckgOp.getLoc()),
        ckgOp.getLoc());
    gatedClockLowerings.insert({clock, info});
    return info;
  }

  // Create the `ClockTreeOp` that corresponds to this ungated clock.
  auto &slot = clockLowerings[clock];
  if (!slot) {
    auto newClock =
        clockBuilder.createOrFold<seq::FromClockOp>(clock.getLoc(), clock);

    // Detect a rising edge on the clock, as `(old != new) & new`.
    auto oldClockStorage = stateBuilder.create<AllocStateOp>(
        clock.getLoc(), StateType::get(stateBuilder.getI1Type()), storageArg);
    auto oldClock =
        clockBuilder.create<StateReadOp>(clock.getLoc(), oldClockStorage);
    clockBuilder.create<StateWriteOp>(clock.getLoc(), oldClockStorage, newClock,
                                      Value{});
    Value trigger = clockBuilder.create<comb::ICmpOp>(
        clock.getLoc(), comb::ICmpPredicate::ne, oldClock, newClock);
    trigger =
        clockBuilder.create<comb::AndOp>(clock.getLoc(), trigger, newClock);

    // Create the tree op.
    auto treeOp = clockBuilder.create<ClockTreeOp>(clock.getLoc(), trigger);
    treeOp.getBody().emplaceBlock();
    slot = std::make_unique<ClockLowering>(clock, treeOp, stats);
  }
  return GatedClockLowering{*slot, Value{}};
}

ClockLowering &ModuleLowering::getOrCreatePassThrough() {
  auto &slot = clockLowerings[Value{}];
  if (!slot) {
    auto treeOp = clockBuilder.create<PassThroughOp>(moduleOp.getLoc());
    treeOp.getBody().emplaceBlock();
    slot = std::make_unique<ClockLowering>(Value{}, treeOp, stats);
  }
  return *slot;
}

ClockLowering &ModuleLowering::getOrCreateInitial() {
  if (!initialLowering) {
    auto treeOp = clockBuilder.create<InitialOp>(moduleOp.getLoc());
    treeOp.getBody().emplaceBlock();
    initialLowering = std::make_unique<ClockLowering>(Value{}, treeOp, stats);
  }
  return *initialLowering;
}

/// Replace all uses of a value with a `StateReadOp` on a state.
Value ModuleLowering::replaceValueWithStateRead(Value value, Value state) {
  OpBuilder builder(state.getContext());
  builder.setInsertionPointAfterValue(state);
  Value readOp = builder.create<StateReadOp>(value.getLoc(), state);
  if (isa<seq::ClockType>(value.getType()))
    readOp = builder.createOrFold<seq::ToClockOp>(value.getLoc(), readOp);
  value.replaceAllUsesWith(readOp);
  return readOp;
}

/// Add the global state as an argument to the module's body block.
void ModuleLowering::addStorageArg() {
  assert(!storageArg);
  storageArg = moduleOp.getBodyBlock()->addArgument(
      StorageType::get(context, {}), moduleOp.getLoc());
}

/// Lower the primary inputs of the module to dedicated ops that allocate the
/// inputs in the model's storage.
LogicalResult ModuleLowering::lowerPrimaryInputs() {
  for (auto blockArg : moduleOp.getBodyBlock()->getArguments()) {
    if (blockArg == storageArg)
      continue;
    auto name = moduleOp.getArgName(blockArg.getArgNumber());
    auto argTy = blockArg.getType();
    IntegerType innerTy;
    if (isa<seq::ClockType>(argTy)) {
      innerTy = IntegerType::get(context, 1);
    } else if (auto intType = dyn_cast<IntegerType>(argTy)) {
      innerTy = intType;
    } else {
      return mlir::emitError(blockArg.getLoc(), "input ")
             << name << " is of non-integer type " << blockArg.getType();
    }
    auto state = stateBuilder.create<RootInputOp>(
        blockArg.getLoc(), StateType::get(innerTy), name, storageArg);
    replaceValueWithStateRead(blockArg, state);
  }
  return success();
}

/// Lower the primary outputs of the module to dedicated ops that allocate the
/// outputs in the model's storage.
LogicalResult ModuleLowering::lowerPrimaryOutputs() {
  auto outputOp = cast<hw::OutputOp>(moduleOp.getBodyBlock()->getTerminator());
  if (outputOp.getNumOperands() > 0) {
    auto outputOperands = SmallVector<Value>(outputOp.getOperands());
    outputOp->dropAllReferences();
    auto &passThrough = getOrCreatePassThrough();
    for (auto [outputArg, name] :
         llvm::zip(outputOperands, moduleOp.getOutputNames())) {
      IntegerType innerTy;
      if (isa<seq::ClockType>(outputArg.getType())) {
        innerTy = IntegerType::get(context, 1);
      } else if (auto intType = dyn_cast<IntegerType>(outputArg.getType())) {
        innerTy = intType;
      } else {
        return mlir::emitError(outputOp.getLoc(), "output ")
               << name << " is of non-integer type " << outputArg.getType();
      }
      auto value = passThrough.materializeValue(outputArg);
      auto state = stateBuilder.create<RootOutputOp>(
          outputOp.getLoc(), StateType::get(innerTy), cast<StringAttr>(name),
          storageArg);
      if (isa<seq::ClockType>(value.getType()))
        value = passThrough.builder.createOrFold<seq::FromClockOp>(
            outputOp.getLoc(), value);
      passThrough.builder.create<StateWriteOp>(outputOp.getLoc(), state, value,
                                               Value{});
    }
  }
  outputOp.erase();
  return success();
}

LogicalResult ModuleLowering::lowerStates() {
  SmallVector<Operation *> opsToLower;
  for (auto &op : *moduleOp.getBodyBlock())
    if (isa<StateOp, MemoryOp, MemoryWritePortOp, TapOp, sim::DPICallOp>(&op))
      opsToLower.push_back(&op);

  for (auto *op : opsToLower) {
    LLVM_DEBUG(llvm::dbgs() << "- Lowering " << *op << "\n");
    auto result =
        TypeSwitch<Operation *, LogicalResult>(op)
            .Case<StateOp, MemoryOp, MemoryWritePortOp, TapOp, sim::DPICallOp>(
                [&](auto op) { return lowerState(op); })
            .Default(success());
    if (failed(result))
      return failure();
  }
  return success();
}

template <typename CallOpTy>
LogicalResult ModuleLowering::lowerStateLike(
    Operation *stateOp, Value stateClock, Value stateEnable, Value stateReset,
    ArrayRef<Value> stateInputs, FlatSymbolRefAttr callee) {
  // Grab all operands from the state op at the callsite and make it drop all
  // its references. This allows `materializeValue` to move an operation if this
  // state was the last user.

  // Get the clock tree and enable condition for this state's clock. If this arc
  // carries an explicit enable condition, fold that into the enable provided by
  // the clock gates in the arc's clock tree.
  auto info = getOrCreateClockLowering(stateClock);
  info.enable = info.clock.getOrCreateAnd(
      info.enable, info.clock.materializeValue(stateEnable), stateOp->getLoc());

  // Allocate the necessary state within the model.
  SmallVector<Value> allocatedStates;
  for (unsigned stateIdx = 0; stateIdx < stateOp->getNumResults(); ++stateIdx) {
    auto type = stateOp->getResult(stateIdx).getType();
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType)
      return stateOp->emitOpError("result ")
             << stateIdx << " has non-integer type " << type
             << "; only integer types are supported";
    auto stateType = StateType::get(intType);
    auto state = stateBuilder.create<AllocStateOp>(stateOp->getLoc(), stateType,
                                                   storageArg);
    if (auto names = stateOp->getAttrOfType<ArrayAttr>("names"))
      state->setAttr("name", names[stateIdx]);
    allocatedStates.push_back(state);
  }

  // Create a copy of the arc use with latency zero. This will effectively be
  // the computation of the arc's transfer function, while the latency is
  // implemented through read and write functions.
  SmallVector<Value> materializedOperands;
  materializedOperands.reserve(stateInputs.size());

  for (auto input : stateInputs)
    materializedOperands.push_back(info.clock.materializeValue(input));

  OpBuilder nonResetBuilder = info.clock.builder;
  if (stateReset) {
    auto materializedReset = info.clock.materializeValue(stateReset);
    auto ifOp = info.clock.builder.create<scf::IfOp>(stateOp->getLoc(),
                                                     materializedReset, true);

    for (auto [alloc, resTy] :
         llvm::zip(allocatedStates, stateOp->getResultTypes())) {
      if (!isa<IntegerType>(resTy))
        stateOp->emitOpError("Non-integer result not supported yet!");

      auto thenBuilder = ifOp.getThenBodyBuilder();
      Value constZero =
          thenBuilder.create<hw::ConstantOp>(stateOp->getLoc(), resTy, 0);
      thenBuilder.create<StateWriteOp>(stateOp->getLoc(), alloc, constZero,
                                       Value());
    }

    nonResetBuilder = ifOp.getElseBodyBuilder();
  }

  stateOp->dropAllReferences();

  auto newStateOp = nonResetBuilder.create<CallOpTy>(
      stateOp->getLoc(), stateOp->getResultTypes(), callee,
      materializedOperands);

  // Create the write ops that write the result of the transfer function to the
  // allocated state storage.
  for (auto [alloc, result] :
       llvm::zip(allocatedStates, newStateOp.getResults()))
    nonResetBuilder.create<StateWriteOp>(stateOp->getLoc(), alloc, result,
                                         info.enable);

  // Replace all uses of the arc with reads from the allocated state.
  for (auto [alloc, result] : llvm::zip(allocatedStates, stateOp->getResults()))
    replaceValueWithStateRead(result, alloc);
  stateOp->erase();
  return success();
}

LogicalResult ModuleLowering::lowerState(StateOp stateOp) {
  // We don't support arcs beyond latency 1 yet. These should be easy to add in
  // the future though.
  if (stateOp.getLatency() > 1)
    return stateOp.emitError("state with latency > 1 not supported");

  auto stateInputs = SmallVector<Value>(stateOp.getInputs());

  return lowerStateLike<arc::CallOp>(stateOp, stateOp.getClock(),
                                     stateOp.getEnable(), stateOp.getReset(),
                                     stateInputs, stateOp.getArcAttr());
}

LogicalResult ModuleLowering::lowerState(sim::DPICallOp callOp) {
  // Clocked call op can be considered as arc state with single latency.
  auto stateClock = callOp.getClock();
  if (!stateClock)
    return callOp.emitError("unclocked DPI call not implemented yet");

  auto stateInputs = SmallVector<Value>(callOp.getInputs());

  return lowerStateLike<func::CallOp>(callOp, stateClock, callOp.getEnable(),
                                      Value(), stateInputs,
                                      callOp.getCalleeAttr());
}

LogicalResult ModuleLowering::lowerState(MemoryOp memOp) {
  auto allocMemOp = stateBuilder.create<AllocMemoryOp>(
      memOp.getLoc(), memOp.getType(), storageArg, memOp->getAttrs());
  memOp.replaceAllUsesWith(allocMemOp.getResult());
  memOp.erase();
  return success();
}

LogicalResult ModuleLowering::lowerState(MemoryWritePortOp memWriteOp) {
  if (memWriteOp.getLatency() > 1)
    return memWriteOp->emitOpError("latencies > 1 not supported yet");

  // Get the clock tree and enable condition for this write port's clock. If the
  // port carries an explicit enable condition, fold that into the enable
  // provided by the clock gates in the port's clock tree.
  auto info = getOrCreateClockLowering(memWriteOp.getClock());

  // Grab all operands from the op and make it drop all its references. This
  // allows `materializeValue` to move an operation if this op was the last
  // user.
  auto writeMemory = memWriteOp.getMemory();
  auto writeInputs = SmallVector<Value>(memWriteOp.getInputs());
  auto arcResultTypes = memWriteOp.getArcResultTypes();
  memWriteOp->dropAllReferences();

  SmallVector<Value> materializedInputs;
  for (auto input : writeInputs)
    materializedInputs.push_back(info.clock.materializeValue(input));
  ValueRange results =
      info.clock.builder
          .create<CallOp>(memWriteOp.getLoc(), arcResultTypes,
                          memWriteOp.getArc(), materializedInputs)
          ->getResults();

  auto enable =
      memWriteOp.getEnable() ? results[memWriteOp.getEnableIdx()] : Value();
  info.enable =
      info.clock.getOrCreateAnd(info.enable, enable, memWriteOp.getLoc());

  // Materialize the operands for the write op within the surrounding clock
  // tree.
  auto address = results[memWriteOp.getAddressIdx()];
  auto data = results[memWriteOp.getDataIdx()];
  if (memWriteOp.getMask()) {
    Value mask = results[memWriteOp.getMaskIdx(static_cast<bool>(enable))];
    Value oldData = info.clock.builder.create<arc::MemoryReadOp>(
        mask.getLoc(), data.getType(), writeMemory, address);
    Value allOnes = info.clock.builder.create<hw::ConstantOp>(
        mask.getLoc(), oldData.getType(), -1);
    Value negatedMask = info.clock.builder.create<comb::XorOp>(
        mask.getLoc(), mask, allOnes, true);
    Value maskedOldData = info.clock.builder.create<comb::AndOp>(
        mask.getLoc(), negatedMask, oldData, true);
    Value maskedNewData =
        info.clock.builder.create<comb::AndOp>(mask.getLoc(), mask, data, true);
    data = info.clock.builder.create<comb::OrOp>(mask.getLoc(), maskedOldData,
                                                 maskedNewData, true);
  }
  info.clock.builder.create<MemoryWriteOp>(memWriteOp.getLoc(), writeMemory,
                                           address, info.enable, data);
  memWriteOp.erase();
  return success();
}

// Add state for taps into the passthrough block.
LogicalResult ModuleLowering::lowerState(TapOp tapOp) {
  auto intType = dyn_cast<IntegerType>(tapOp.getValue().getType());
  if (!intType)
    return mlir::emitError(tapOp.getLoc(), "tapped value ")
           << tapOp.getNameAttr() << " is of non-integer type "
           << tapOp.getValue().getType();

  // Grab what we need from the tap op and then make it drop all its references.
  // This will allow `materializeValue` to move ops instead of cloning them.
  auto tapValue = tapOp.getValue();
  tapOp->dropAllReferences();

  auto &passThrough = getOrCreatePassThrough();
  auto materializedValue = passThrough.materializeValue(tapValue);
  auto state = stateBuilder.create<AllocStateOp>(
      tapOp.getLoc(), StateType::get(intType), storageArg, true);
  state->setAttr("name", tapOp.getNameAttr());
  passThrough.builder.create<StateWriteOp>(tapOp.getLoc(), state,
                                           materializedValue, Value{});
  tapOp.erase();
  return success();
}

/// Lower all instances of external modules to internal inputs/outputs to be
/// driven from outside of the design.
LogicalResult ModuleLowering::lowerExtModules(SymbolTable &symtbl) {
  auto instOps = SmallVector<InstanceOp>(moduleOp.getOps<InstanceOp>());
  for (auto op : instOps)
    if (isa<HWModuleExternOp>(symtbl.lookup(op.getModuleNameAttr().getAttr())))
      if (failed(lowerExtModule(op)))
        return failure();
  return success();
}

LogicalResult ModuleLowering::lowerExtModule(InstanceOp instOp) {
  LLVM_DEBUG(llvm::dbgs() << "- Lowering extmodule "
                          << instOp.getInstanceNameAttr() << "\n");

  SmallString<32> baseName(instOp.getInstanceName());
  auto baseNameLen = baseName.size();

  // Lower the inputs of the extmodule as state that is only written.
  for (auto [operand, name] :
       llvm::zip(instOp.getOperands(), instOp.getArgNames())) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Input " << name << " : " << operand.getType() << "\n");
    auto intType = dyn_cast<IntegerType>(operand.getType());
    if (!intType)
      return mlir::emitError(operand.getLoc(), "input ")
             << name << " of extern module " << instOp.getModuleNameAttr()
             << " instance " << instOp.getInstanceNameAttr()
             << " is of non-integer type " << operand.getType();
    baseName.resize(baseNameLen);
    baseName += '/';
    baseName += cast<StringAttr>(name).getValue();
    auto &passThrough = getOrCreatePassThrough();
    auto state = stateBuilder.create<AllocStateOp>(
        instOp.getLoc(), StateType::get(intType), storageArg);
    state->setAttr("name", stateBuilder.getStringAttr(baseName));
    passThrough.builder.create<StateWriteOp>(
        instOp.getLoc(), state, passThrough.materializeValue(operand), Value{});
  }

  // Lower the outputs of the extmodule as state that is only read.
  for (auto [result, name] :
       llvm::zip(instOp.getResults(), instOp.getResultNames())) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Output " << name << " : " << result.getType() << "\n");
    auto intType = dyn_cast<IntegerType>(result.getType());
    if (!intType)
      return mlir::emitError(result.getLoc(), "output ")
             << name << " of extern module " << instOp.getModuleNameAttr()
             << " instance " << instOp.getInstanceNameAttr()
             << " is of non-integer type " << result.getType();
    baseName.resize(baseNameLen);
    baseName += '/';
    baseName += cast<StringAttr>(name).getValue();
    auto state = stateBuilder.create<AllocStateOp>(
        result.getLoc(), StateType::get(intType), storageArg);
    state->setAttr("name", stateBuilder.getStringAttr(baseName));
    replaceValueWithStateRead(result, state);
  }

  instOp.erase();
  return success();
}

LogicalResult ModuleLowering::cleanup() {
  // Clean up dead ops in the model.
  SetVector<Operation *> erasureWorklist;
  auto isDead = [](Operation *op) {
    if (isOpTriviallyDead(op))
      return true;
    if (!op->use_empty())
      return false;
    return false;
  };
  for (auto &op : *moduleOp.getBodyBlock())
    if (isDead(&op))
      erasureWorklist.insert(&op);
  while (!erasureWorklist.empty()) {
    auto *op = erasureWorklist.pop_back_val();
    if (!isDead(op))
      continue;
    op->walk([&](Operation *innerOp) {
      for (auto operand : innerOp->getOperands())
        if (auto *defOp = operand.getDefiningOp())
          if (!op->isProperAncestor(defOp))
            erasureWorklist.insert(defOp);
    });
    op->erase();
  }

  // Establish an order among all operations (to avoid an O(n²) pathological
  // pattern with `moveBefore`) and replicate read operations into the blocks
  // where they have uses. The established order is used to create the read
  // operation as late in the block as possible, just before the first use.
  DenseMap<Operation *, unsigned> opOrder;
  SmallVector<StateReadOp, 0> readsToSink;
  moduleOp.walk([&](Operation *op) {
    opOrder.insert({op, opOrder.size()});
    if (auto readOp = dyn_cast<StateReadOp>(op))
      readsToSink.push_back(readOp);
  });
  for (auto readToSink : readsToSink) {
    SmallDenseMap<Block *, std::pair<StateReadOp, unsigned>> readsByBlock;
    for (auto &use : llvm::make_early_inc_range(readToSink->getUses())) {
      auto *user = use.getOwner();
      auto userOrder = opOrder.lookup(user);
      auto &localRead = readsByBlock[user->getBlock()];
      if (!localRead.first) {
        if (user->getBlock() == readToSink->getBlock()) {
          localRead.first = readToSink;
          readToSink->moveBefore(user);
        } else {
          localRead.first = OpBuilder(user).cloneWithoutRegions(readToSink);
        }
        localRead.second = userOrder;
      } else if (userOrder < localRead.second) {
        localRead.first->moveBefore(user);
        localRead.second = userOrder;
      }
      use.set(localRead.first);
    }
    if (readToSink.use_empty())
      readToSink.erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerStatePass : public arc::impl::LowerStateBase<LowerStatePass> {
  LowerStatePass() = default;
  LowerStatePass(const LowerStatePass &pass) : LowerStatePass() {}

  void runOnOperation() override;
  LogicalResult runOnModule(HWModuleOp moduleOp, SymbolTable &symtbl);

  Statistics stats{this};
};
} // namespace

void LowerStatePass::runOnOperation() {
  auto &symtbl = getAnalysis<SymbolTable>();
  SmallVector<HWModuleExternOp> extModules;
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    if (auto moduleOp = dyn_cast<HWModuleOp>(&op)) {
      if (failed(runOnModule(moduleOp, symtbl)))
        return signalPassFailure();
    } else if (auto extModuleOp = dyn_cast<HWModuleExternOp>(&op)) {
      extModules.push_back(extModuleOp);
    }
  }
  for (auto op : extModules)
    op.erase();

  // Lower remaining MemoryReadPort ops to MemoryRead ops. This can occur when
  // the fan-in of a MemoryReadPortOp contains another such operation and is
  // materialized before the one in the fan-in as the MemoryReadPortOp is not
  // marked as a fan-in blocking/termination operation in `shouldMaterialize`.
  // Adding it there can lead to dominance issues which would then have to be
  // resolved instead.
  SetVector<DefineOp> arcsToLower;
  OpBuilder builder(getOperation());
  getOperation()->walk([&](MemoryReadPortOp memReadOp) {
    if (auto defOp = memReadOp->getParentOfType<DefineOp>())
      arcsToLower.insert(defOp);

    builder.setInsertionPoint(memReadOp);
    Value newRead = builder.create<MemoryReadOp>(
        memReadOp.getLoc(), memReadOp.getMemory(), memReadOp.getAddress());
    memReadOp.replaceAllUsesWith(newRead);
    memReadOp.erase();
  });

  SymbolTableCollection symbolTable;
  mlir::SymbolUserMap userMap(symbolTable, getOperation());
  for (auto defOp : arcsToLower) {
    auto *terminator = defOp.getBodyBlock().getTerminator();
    builder.setInsertionPoint(terminator);
    builder.create<func::ReturnOp>(terminator->getLoc(),
                                   terminator->getOperands());
    terminator->erase();
    builder.setInsertionPoint(defOp);
    auto funcOp = builder.create<func::FuncOp>(defOp.getLoc(), defOp.getName(),
                                               defOp.getFunctionType());
    funcOp->setAttr("llvm.linkage",
                    LLVM::LinkageAttr::get(builder.getContext(),
                                           LLVM::linkage::Linkage::Internal));
    funcOp.getBody().takeBody(defOp.getBody());

    for (auto *user : userMap.getUsers(defOp)) {
      builder.setInsertionPoint(user);
      ValueRange results = builder
                               .create<func::CallOp>(
                                   user->getLoc(), funcOp,
                                   cast<CallOpInterface>(user).getArgOperands())
                               ->getResults();
      user->replaceAllUsesWith(results);
      user->erase();
    }

    defOp.erase();
  }
}

LogicalResult LowerStatePass::runOnModule(HWModuleOp moduleOp,
                                          SymbolTable &symtbl) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering state in `" << moduleOp.getModuleName()
                          << "`\n");
  ModuleLowering lowering(moduleOp, stats);

  // Add sentinel ops to separate state allocations from clock trees.
  lowering.stateBuilder.setInsertionPointToStart(moduleOp.getBodyBlock());

  Operation *stateSentinel =
      lowering.stateBuilder.create<hw::OutputOp>(moduleOp.getLoc());
  Operation *clockSentinel =
      lowering.stateBuilder.create<hw::OutputOp>(moduleOp.getLoc());

  lowering.stateBuilder.setInsertionPoint(stateSentinel);
  lowering.clockBuilder.setInsertionPoint(clockSentinel);

  lowering.addStorageArg();
  if (failed(lowering.lowerPrimaryInputs()))
    return failure();
  if (failed(lowering.lowerPrimaryOutputs()))
    return failure();
  if (failed(lowering.lowerStates()))
    return failure();
  if (failed(lowering.lowerExtModules(symtbl)))
    return failure();

  // Clean up the module body which contains a lot of operations that the
  // pessimistic value materialization has left behind because it couldn't
  // reliably determine that the ops were no longer needed.
  if (failed(lowering.cleanup()))
    return failure();

  // Erase the sentinel ops.
  stateSentinel->erase();
  clockSentinel->erase();

  // Replace the `HWModuleOp` with a `ModelOp`.
  moduleOp.getBodyBlock()->eraseArguments(
      [&](auto arg) { return arg != lowering.storageArg; });
  ImplicitLocOpBuilder builder(moduleOp.getLoc(), moduleOp);
  auto modelOp =
      builder.create<ModelOp>(moduleOp.getLoc(), moduleOp.getModuleNameAttr(),
                              TypeAttr::get(moduleOp.getModuleType()));
  modelOp.getBody().takeBody(moduleOp.getBody());
  moduleOp->erase();
  sortTopologically(&modelOp.getBodyBlock());

  return success();
}

std::unique_ptr<Pass> arc::createLowerStatePass() {
  return std::make_unique<LowerStatePass>();
}
