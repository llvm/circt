//===- LowerState.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombDialect.h"
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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

namespace {
enum class Phase { Initial, Old, New, Final };

template <class OS>
OS &operator<<(OS &os, Phase phase) {
  switch (phase) {
  case Phase::Initial:
    return os << "Initial";
  case Phase::Old:
    return os << "Old";
  case Phase::New:
    return os << "New";
  case Phase::Final:
    return os << "Final";
  }
}

struct ModuleLowering;

struct OpLowering {
  Operation *op;
  Phase phase;
  ModuleLowering &module;

  bool initial = true;
  SmallVector<std::pair<Operation *, Phase>, 2> pending;

  OpLowering(Operation *op, Phase phase, ModuleLowering &module)
      : op(op), phase(phase), module(module) {}
  LogicalResult lower();
  LogicalResult lower(StateOp op);
  LogicalResult lower(hw::OutputOp op);
  LogicalResult lowerDefault();
  Value lowerValue(Value value, Phase phase);
  void addPending(Value value, Phase phase);
  void addPending(Operation *op, Phase phase);
};

/// All state associated with lowering a single module.
struct ModuleLowering {
  /// The module being lowered.
  HWModuleOp moduleOp;
  /// The builder for the main body of the model.
  OpBuilder builder;
  /// The storage value that can be used for `arc.alloc_state` and friends.
  Value storageArg;
  /// A worklist of pending op lowerings.
  SmallVector<OpLowering> opsWorklist;
  /// The set of ops currently in the worklist. Used to detect cycles.
  SmallDenseSet<std::pair<Operation *, Phase>> opsSeen;
  /// The allocated input ports.
  SmallVector<Value> allocatedInputs;
  /// The allocated states as a mapping from op results to `arc.alloc_state`
  /// results.
  DenseMap<Value, Value> allocatedStates;
  /// The ops that have already been lowered.
  DenseSet<std::pair<Operation *, Phase>> loweredOps;
  /// The values that have already been lowered.
  DenseMap<std::pair<Value, Phase>, Value> loweredValues;
  /// A mapping from unlowered clocks to a value indicating a posedge. This is
  /// used to not create an excessive number of posedge detectors.
  DenseMap<Value, Value> loweredPosedges;
  /// The previous enable and the value it was lowered to. This is used to reuse
  /// previous if ops for the same enable value.
  std::pair<Value, Value> prevEnable;
  /// The previous reset and the value it was lowered to. This is used to reuse
  /// previous uf ops for the same reset value.
  std::pair<Value, Value> prevReset;

  ModuleLowering(HWModuleOp moduleOp) : moduleOp(moduleOp), builder(moduleOp) {}
  LogicalResult run();
  LogicalResult lowerOp(Operation *op);
  Value getAllocatedState(OpResult result);
  Value detectPosedge(Value clock);
};
} // namespace

//===----------------------------------------------------------------------===//
// Module Lowering
//===----------------------------------------------------------------------===//

LogicalResult ModuleLowering::run() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering module `" << moduleOp.getModuleName()
                          << "`\n");

  // Create the replacement `ModelOp`.
  auto modelOp = builder.create<ModelOp>(
      moduleOp.getLoc(), moduleOp.getModuleNameAttr(),
      TypeAttr::get(moduleOp.getModuleType()), FlatSymbolRefAttr{});
  auto &modelBlock = modelOp.getBody().emplaceBlock();
  storageArg = modelBlock.addArgument(
      StorageType::get(builder.getContext(), {}), modelOp.getLoc());
  builder.setInsertionPointToStart(&modelBlock);

  // Allocate storage for the inputs.
  for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
    auto name = moduleOp.getArgName(arg.getArgNumber());
    auto type = dyn_cast<IntegerType>(arg.getType());
    if (!type && isa<seq::ClockType>(arg.getType()))
      type = builder.getI1Type();
    if (!type)
      return mlir::emitError(arg.getLoc(), "input ")
             << name << " is of non-integer type " << arg.getType();
    auto state = builder.create<RootInputOp>(arg.getLoc(), StateType::get(type),
                                             name, storageArg);
    allocatedInputs.push_back(state);
  }

  // Lower the ops.
  for (auto &op : moduleOp.getOps()) {
    if (mlir::isMemoryEffectFree(&op) && !isa<hw::OutputOp>(op))
      continue;
    if (failed(lowerOp(&op)))
      return failure();
  }

  // Clean up any dead ops. The lowering inserts a few defensive
  // `arc.state_read` ops that may remain unused. This cleans them up.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(modelBlock)))
    if (mlir::isOpTriviallyDead(&op))
      op.erase();

  return success();
}

/// Lower an op and its entire fan-in cone.
LogicalResult ModuleLowering::lowerOp(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "- Handling " << *op << "\n");
  auto phase = Phase::New;
  opsWorklist.push_back(OpLowering(op, phase, *this));
  opsSeen.insert({op, phase});

  while (!opsWorklist.empty()) {
    auto &opLowering = opsWorklist.back();

    // Collect an initial list of operands that need to be lowered.
    if (opLowering.initial) {
      if (failed(opLowering.lower()))
        return failure();
      std::reverse(opLowering.pending.begin(), opLowering.pending.end());
      opLowering.initial = false;
      if (!opLowering.pending.empty())
        LLVM_DEBUG(llvm::dbgs() << "  - Found " << opLowering.pending.size()
                                << " dependencies for " << opLowering.phase
                                << " " << *opLowering.op << "\n");
    }

    // Push operands onto the worklist.
    if (!opLowering.pending.empty()) {
      auto [defOp, phase] = opLowering.pending.pop_back_val();
      if (loweredOps.contains({defOp, phase}))
        continue;
      if (!opsSeen.insert({defOp, phase}).second) {
        defOp->emitOpError("is on a combinational loop");
        return failure();
      }
      opsWorklist.push_back(OpLowering(defOp, phase, *this));
      continue;
    }

    // At this point all operands are available and the op itself can be
    // lowered.
    LLVM_DEBUG(llvm::dbgs() << "  - Lowering " << opLowering.phase << " "
                            << *opLowering.op << "\n");
    if (failed(opLowering.lower()))
      return failure();
    loweredOps.insert({opLowering.op, opLowering.phase});
    opsSeen.erase({opLowering.op, opLowering.phase});
    opsWorklist.pop_back();
  }

  return success();
}

/// Return the `arc.alloc_state` associated with the given state op result.
/// Creates the allocation op if it does not yet exist.
Value ModuleLowering::getAllocatedState(OpResult result) {
  if (auto alloc = allocatedStates.lookup(result))
    return alloc;

  // Zip up to the body of the `ModelOp` we're currently populating and create
  // the allocation op there.
  OpBuilder::InsertionGuard guard(builder);
  while (builder.getBlock() != storageArg.getParentBlock())
    builder.setInsertionPoint(builder.getBlock()->getParentOp());

  auto type = dyn_cast<IntegerType>(result.getType());
  if (!type) {
    result.getOwner()->emitOpError()
        << "result #" << result.getResultNumber() << " is of non-integer type "
        << result.getType();
    return {};
  }
  auto alloc = builder.create<AllocStateOp>(result.getLoc(),
                                            StateType::get(type), storageArg);
  allocatedStates.insert({result, alloc});

  // HACK: If the result comes from an op that has a "names" attribute, use that
  // as a name for the allocation. This should no longer be necessary once we
  // properly support the Debug dialect.
  if (auto names = result.getOwner()->getAttrOfType<ArrayAttr>("names"))
    if (names.size() > result.getResultNumber())
      alloc->setAttr("name", names[result.getResultNumber()]);

  return alloc;
}

/// Allocate the necessary storage, reads, writes, and comparisons to detect a
/// rising edge on a clock value.
Value ModuleLowering::detectPosedge(Value clock) {
  auto loc = clock.getLoc();
  if (isa<seq::ClockType>(clock.getType()))
    clock = builder.createOrFold<seq::FromClockOp>(loc, clock);

  // Allocate storage to store the previous clock value.
  auto oldStorage = builder.create<AllocStateOp>(
      loc, StateType::get(builder.getI1Type()), storageArg);

  // Read the old clock value from storage and write the new clock value to
  // storage.
  auto oldClock = builder.create<StateReadOp>(loc, oldStorage);
  builder.create<StateWriteOp>(loc, oldStorage, clock, Value{});

  // Detect a rising edge.
  Value edge = builder.create<comb::XorOp>(loc, oldClock, clock);
  edge = builder.create<comb::AndOp>(loc, edge, clock);
  return edge;
}

//===----------------------------------------------------------------------===//
// Operation Lowering
//===----------------------------------------------------------------------===//

LogicalResult OpLowering::lower() {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<StateOp, hw::OutputOp>([&](auto op) { return lower(op); })
      .Case<InstanceOp, TapOp, MemoryOp, MemoryReadPortOp, MemoryWritePortOp,
            sim::DPICallOp, seq::InitialOp>([&](auto op) {
        if (initial)
          return success();
        op.emitOpError() << "state lowering not yet implemented";
        return failure();
      })
      .Default([&](auto) { return lowerDefault(); });
}

LogicalResult OpLowering::lowerDefault() {
  if (op->getNumRegions() > 0)
    return op->emitOpError("has regions which is not supported in LowerState");

  // Lower the operand values.
  SmallVector<Value> values;
  for (auto operand : op->getOperands())
    values.push_back(lowerValue(operand, phase));
  if (initial)
    return success();
  if (llvm::is_contained(values, Value{}))
    return failure();

  // Clone the operation.
  auto *clonedOp = module.builder.clone(*op);
  clonedOp->setOperands(values);

  // Keep track of the results.
  for (auto [result, lowered] :
       llvm::zip(op->getResults(), clonedOp->getResults()))
    module.loweredValues[{result, phase}] = lowered;

  return success();
}

LogicalResult OpLowering::lower(StateOp op) {
  if (phase != Phase::New)
    return op.emitOpError() << "cannot be lowered in the " << phase << " phase";

  // Ensure all operands are lowered before we lower the op itself. State ops
  // are special in that they require the "old" value of their inputs and
  // enable, in order to compute the updated "new" value. The clock needs to be
  // the "new" value though, such that other states can act as a clock source.
  if (initial) {
    if (op.getClock())
      lowerValue(op.getClock(), Phase::New);
    if (op.getEnable())
      lowerValue(op.getEnable(), Phase::Old);
    for (auto input : op.getInputs())
      lowerValue(input, Phase::Old);
    return success();
  }

  if (!op.getClock())
    return op.emitOpError() << "must have a clock";
  if (!op.getInitials().empty())
    return op.emitOpError() << "must not have initial values";

  // Detect the clock edge.
  auto &posedge = module.loweredPosedges[op.getClock()];
  if (!posedge) {
    auto clock = lowerValue(op.getClock(), Phase::New);
    if (!clock)
      return failure();
    posedge = module.detectPosedge(clock);
  }

  // Check if we're inserting right after an if op for the same clock edge, in
  // which case we can reuse that op. Otherwise create the new if op.
  scf::IfOp ifClockOp;
  if (auto ip = module.builder.getInsertionPoint();
      ip != module.builder.getBlock()->begin())
    if (auto ifOp = dyn_cast<scf::IfOp>(*std::prev(ip)))
      if (ifOp.getCondition() == posedge)
        ifClockOp = ifOp;
  if (!ifClockOp)
    ifClockOp =
        module.builder.create<scf::IfOp>(posedge.getLoc(), posedge, false);
  OpBuilder::InsertionGuard guard(module.builder);
  module.builder.setInsertionPoint(ifClockOp.thenYield());

  // Make sure we have the state storage available such that we can read and
  // write from and to them.
  SmallVector<Value> states;
  for (auto result : op.getResults()) {
    auto state = module.getAllocatedState(result);
    if (!state)
      return failure();
    states.push_back(state);
  }

  // Handle the reset.
  if (op.getReset()) {
    // Check if we can reuse a previous reset value.
    auto &[unloweredReset, reset] = module.prevReset;
    if (unloweredReset != op.getReset() ||
        reset.getParentBlock() != module.builder.getBlock()) {
      unloweredReset = op.getReset();
      reset = lowerValue(op.getReset(), Phase::Old);
      if (!reset)
        return failure();
    }

    // Check if we're inserting right after an if op for the same reset, in
    // which case we can reuse that op. Otherwise create the new if op.
    scf::IfOp ifResetOp;
    if (auto ip = module.builder.getInsertionPoint();
        ip != module.builder.getBlock()->begin())
      if (auto ifOp = dyn_cast<scf::IfOp>(*std::prev(ip)))
        if (ifOp.getCondition() == reset)
          ifResetOp = ifOp;
    if (!ifResetOp)
      ifResetOp = module.builder.create<scf::IfOp>(op.getLoc(), reset, true);
    module.builder.setInsertionPoint(ifResetOp.thenYield());

    // Generate the zero value writes.
    for (auto state : states) {
      auto type = cast<StateType>(state.getType()).getType();
      Value value = module.builder.create<ConstantOp>(
          reset.getLoc(), module.builder.getIntegerType(hw::getBitWidth(type)),
          0);
      if (value.getType() != type)
        value = module.builder.create<BitcastOp>(reset.getLoc(), type, value);
      module.builder.create<StateWriteOp>(reset.getLoc(), state, value,
                                          Value{});
    }
    module.builder.setInsertionPoint(ifResetOp.elseYield());
  }

  // Handle the enable.
  if (op.getEnable()) {
    // Check if we can reuse a previous enable value.
    auto &[unloweredEnable, enable] = module.prevEnable;
    if (unloweredEnable != op.getEnable() ||
        enable.getParentBlock() != module.builder.getBlock()) {
      unloweredEnable = op.getEnable();
      enable = lowerValue(op.getEnable(), Phase::Old);
      if (!enable)
        return failure();
    }

    // Check if we're inserting right after an if op for the same enable, in
    // which case we can reuse that op. Otherwise create the new if op.
    scf::IfOp ifEnableOp;
    if (auto ip = module.builder.getInsertionPoint();
        ip != module.builder.getBlock()->begin())
      if (auto ifOp = dyn_cast<scf::IfOp>(*std::prev(ip)))
        if (ifOp.getCondition() == enable)
          ifEnableOp = ifOp;
    if (!ifEnableOp)
      ifEnableOp = module.builder.create<scf::IfOp>(op.getLoc(), enable, false);
    module.builder.setInsertionPoint(ifEnableOp.thenYield());
  }

  // Get the transfer function inputs. This potential inserts read ops.
  SmallVector<Value> inputs;
  for (auto input : op.getInputs()) {
    auto lowered = lowerValue(input, Phase::Old);
    if (!lowered)
      return failure();
    inputs.push_back(lowered);
  }

  // Compute the transfer function and write its results to the state's storage.
  auto callOp = module.builder.create<CallOp>(op.getLoc(), op.getResultTypes(),
                                              op.getArc(), inputs);
  for (auto [state, value] : llvm::zip(states, callOp.getResults()))
    module.builder.create<StateWriteOp>(value.getLoc(), state, value, Value{});

  // Since we just wrote the new state value to storage, insert read ops just
  // before the if op that keep the old value around for any later ops that
  // still need it.
  module.builder.setInsertionPoint(ifClockOp);
  for (auto [state, result] : llvm::zip(states, op.getResults())) {
    auto oldValue = module.builder.create<StateReadOp>(result.getLoc(), state);
    module.loweredValues[{result, Phase::Old}] = oldValue;
  }

  return success();
}

LogicalResult OpLowering::lower(hw::OutputOp op) {
  // First get the current value of all outputs.
  SmallVector<Value> values;
  for (auto operand : op.getOperands())
    values.push_back(lowerValue(operand, Phase::New));
  if (initial)
    return success();
  if (llvm::is_contained(values, Value{}))
    return failure();

  // Then allocate storage for each output and assign the corresponding value.
  for (auto [value, name] :
       llvm::zip(values, module.moduleOp.getOutputNames())) {
    auto loc = value.getLoc(); // should actually be output port location
    auto type = dyn_cast<IntegerType>(value.getType());
    if (!type && isa<seq::ClockType>(value.getType()))
      type = module.builder.getI1Type();
    if (!type)
      return mlir::emitError(loc, "output ")
             << name << " is of non-integer type " << value.getType();
    auto state = module.builder.create<RootOutputOp>(
        loc, StateType::get(type), cast<StringAttr>(name), module.storageArg);
    auto castValue = value;
    if (isa<seq::ClockType>(value.getType()))
      castValue =
          module.builder.create<seq::FromClockOp>(value.getLoc(), castValue);
    module.builder.create<StateWriteOp>(value.getLoc(), state, castValue,
                                        Value{});
  }
  return success();
}

Value OpLowering::lowerValue(Value value, Phase phase) {
  // Handle module inputs. They read the same in all phases.
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (initial)
      return {};
    auto state = module.allocatedInputs[arg.getArgNumber()];
    auto read = module.builder.create<StateReadOp>(arg.getLoc(), state);
    if (isa<seq::ClockType>(arg.getType()))
      return module.builder.create<seq::ToClockOp>(arg.getLoc(), read);
    return read;
  }

  // Check if the value has already been lowered.
  if (auto lowered = module.loweredValues.lookup({value, phase}))
    return lowered;

  // At this point the value is the result of an op. (Block arguments are
  // handled above.)
  auto result = cast<OpResult>(value);
  auto *op = result.getOwner();

  // Special handling for state ops.
  if (isa<StateOp>(op)) {
    // The old value of a state can be read from its allocated storage. The
    // lowering guarantees that the updated value has not yet been written to
    // storage. (If such an update would have already happened, the
    // `loweredValues` lookup above would have returned the old value of the
    // state.)
    if (phase == Phase::Old) {
      if (initial)
        return {};
      assert(!module.loweredOps.contains({op, Phase::New}) &&
             "need old value but new value already written");
      auto state = module.getAllocatedState(result);
      if (!state)
        return {};
      return module.builder.create<StateReadOp>(result.getLoc(), state);
    }

    // To get the new value of a state after it has been updated, first compute
    // the state update itself, and then simply read back the updated value.
    if (phase == Phase::New) {
      if (initial) {
        addPending(op, Phase::New);
        return {};
      }
      auto state = module.getAllocatedState(result);
      if (!state)
        return {};
      return module.builder.create<StateReadOp>(result.getLoc(), state);
    }

    if (!initial)
      op->emitOpError() << "result cannot be used in " << phase << " phase\n";
    return {};
  }

  // Otherwise we first have to lower the given value.
  if (initial) {
    addPending(op, phase);
    return {};
  }
  auto d = op->emitOpError() << "operand value has not been lowered";
  d.attachNote(result.getLoc()) << "value defined here";
  return {};
}

void OpLowering::addPending(Value value, Phase phase) {
  auto *defOp = value.getDefiningOp();
  assert(defOp && "block args should never be marked as a dependency");
  addPending(defOp, phase);
}

void OpLowering::addPending(Operation *op, Phase phase) {
  auto pair = std::make_pair(op, phase);
  if (!module.loweredOps.contains(pair))
    if (!llvm::is_contained(pending, pair))
      pending.push_back(pair);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerStatePass : public arc::impl::LowerStateBase<LowerStatePass> {
  using LowerStateBase::LowerStateBase;
  void runOnOperation() override;
};
} // namespace

void LowerStatePass::runOnOperation() {
  auto op = getOperation();
  for (auto moduleOp : llvm::make_early_inc_range(op.getOps<HWModuleOp>())) {
    if (failed(ModuleLowering(moduleOp).run()))
      return signalPassFailure();
    moduleOp.erase();
  }
  for (auto extModuleOp :
       llvm::make_early_inc_range(op.getOps<HWModuleExternOp>()))
    extModuleOp.erase();
}

std::unique_ptr<Pass> arc::createLowerStatePass() {
  return std::make_unique<LowerStatePass>();
}
