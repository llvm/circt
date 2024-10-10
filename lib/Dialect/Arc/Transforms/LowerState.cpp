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
#define GEN_PASS_DEF_LOWERSTATEPASS
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
    return os << "initial";
  case Phase::Old:
    return os << "old";
  case Phase::New:
    return os << "new";
  case Phase::Final:
    return os << "final";
  }
}

struct ModuleLowering;

/// All state associated with lowering a single operation. Instances of this
/// struct are kept on a worklist to perform a depth-first traversal of the
/// module being lowered.
///
/// The actual lowering occurs in `lower()`. This function is called exactly
/// twice. A first time with `initial` being true, where other values and
/// operations that have to be lowered first may be marked with `addPending`. No
/// actual lowering or error reporting should occur when `initial` is true. The
/// worklist then ensures that all `pending` ops are lowered before `lower()` is
/// called a second time with `initial` being false. At this point the actual
/// lowering and error reporting should occur.
///
/// The `initial` variable is used to allow for a single block of code to mark
/// values and ops as dependencies and actually do the lowering based on them.
struct OpLowering {
  Operation *op;
  Phase phase;
  ModuleLowering &module;

  bool initial = true;
  SmallVector<std::pair<Operation *, Phase>, 2> pending;

  OpLowering(Operation *op, Phase phase, ModuleLowering &module)
      : op(op), phase(phase), module(module) {}

  // Operation Lowering.
  LogicalResult lower();
  LogicalResult lowerDefault();
  LogicalResult lower(StateOp op);
  LogicalResult lower(MemoryOp op);
  LogicalResult lower(TapOp op);
  LogicalResult lower(InstanceOp op);
  LogicalResult lower(hw::OutputOp op);
  LogicalResult lower(seq::InitialOp op);

  scf::IfOp createIfClockOp(Value clock);

  // Value Lowering. These functions are called from the `lower()` functions
  // above. They handle values used by the `op`. This can generate reads from
  // state and memory storage on-the-fly, or mark other ops as dependencies to
  // be lowered first.
  Value lowerValue(Value value, Phase phase);
  Value lowerValue(InstanceOp op, OpResult result, Phase phase);
  Value lowerValue(StateOp op, OpResult result, Phase phase);
  Value lowerValue(MemoryReadPortOp op, OpResult result, Phase phase);
  Value lowerValue(seq::FromImmutableOp op, OpResult result, Phase phase);

  void addPending(Value value, Phase phase);
  void addPending(Operation *op, Phase phase);
};

/// All state associated with lowering a single module.
struct ModuleLowering {
  /// The module being lowered.
  HWModuleOp moduleOp;
  /// The builder for the main body of the model.
  OpBuilder builder;
  /// The builder for state allocation ops.
  OpBuilder allocBuilder;
  /// The builder for the initial phase.
  OpBuilder initialBuilder;
  /// The builder for the final phase.
  OpBuilder finalBuilder;

  /// The storage value that can be used for `arc.alloc_state` and friends.
  Value storageArg;

  /// A worklist of pending op lowerings.
  SmallVector<OpLowering> opsWorklist;
  /// The set of ops currently in the worklist. Used to detect cycles.
  SmallDenseSet<std::pair<Operation *, Phase>> opsSeen;
  /// The ops that have already been lowered.
  DenseSet<std::pair<Operation *, Phase>> loweredOps;
  /// The values that have already been lowered.
  DenseMap<std::pair<Value, Phase>, Value> loweredValues;

  /// The allocated input ports.
  SmallVector<Value> allocatedInputs;
  /// The allocated states as a mapping from op results to `arc.alloc_state`
  /// results.
  DenseMap<Value, Value> allocatedStates;
  /// The allocated storage for values computed during the initial phase.
  DenseMap<Value, Value> allocatedInitials;
  /// The allocated storage for taps.
  DenseMap<Operation *, Value> allocatedTaps;

  /// A mapping from unlowered clocks to a value indicating a posedge. This is
  /// used to not create an excessive number of posedge detectors.
  DenseMap<Value, Value> loweredPosedges;
  /// The previous enable and the value it was lowered to. This is used to reuse
  /// previous if ops for the same enable value.
  std::pair<Value, Value> prevEnable;
  /// The previous reset and the value it was lowered to. This is used to reuse
  /// previous uf ops for the same reset value.
  std::pair<Value, Value> prevReset;

  ModuleLowering(HWModuleOp moduleOp)
      : moduleOp(moduleOp), builder(moduleOp), allocBuilder(moduleOp),
        initialBuilder(moduleOp), finalBuilder(moduleOp) {}
  LogicalResult run();
  LogicalResult lowerOp(Operation *op);
  Value getAllocatedState(OpResult result);
  Value detectPosedge(Value clock);
  OpBuilder &getBuilder(Phase phase);
  Value requireLoweredValue(Value value, Phase phase, Location useLoc);
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

  // Create the `arc.initial` op to contain the ops for the initialization
  // phase.
  auto initialOp = builder.create<InitialOp>(moduleOp.getLoc());
  initialBuilder.setInsertionPointToStart(&initialOp.getBody().emplaceBlock());

  // Create the `arc.final` op to contain the ops for the finalization phase.
  auto finalOp = builder.create<FinalOp>(moduleOp.getLoc());
  finalBuilder.setInsertionPointToStart(&finalOp.getBody().emplaceBlock());

  // Position the alloc builder such that allocation ops get inserted above the
  // initial op.
  allocBuilder.setInsertionPoint(initialOp);

  // Allocate storage for the inputs.
  for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
    auto name = moduleOp.getArgName(arg.getArgNumber());
    auto state = allocBuilder.create<RootInputOp>(
        arg.getLoc(), StateType::get(arg.getType()), name, storageArg);
    allocatedInputs.push_back(state);
  }

  // Lower the ops.
  for (auto &op : moduleOp.getOps()) {
    if (mlir::isMemoryEffectFree(&op) && !isa<hw::OutputOp>(op))
      continue;
    if (isa<MemoryReadPortOp, MemoryWritePortOp>(op))
      continue; // handled as part of `MemoryOp`
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
  if (isa<seq::InitialOp>(op))
    phase = Phase::Initial;
  if (loweredOps.contains({op, phase}))
    return success();
  opsWorklist.push_back(OpLowering(op, phase, *this));
  opsSeen.insert({op, phase});

  auto dumpWorklist = [&] {
    for (auto &opLowering : llvm::reverse(opsWorklist))
      opLowering.op->emitRemark()
          << "used for " << opLowering.phase << " phase here";
  };

  while (!opsWorklist.empty()) {
    auto &opLowering = opsWorklist.back();

    // Collect an initial list of operands that need to be lowered.
    if (opLowering.initial) {
      if (failed(opLowering.lower())) {
        dumpWorklist();
        return failure();
      }
      std::reverse(opLowering.pending.begin(), opLowering.pending.end());
      opLowering.initial = false;
    }

    // Push operands onto the worklist.
    if (!opLowering.pending.empty()) {
      auto [defOp, phase] = opLowering.pending.pop_back_val();
      if (loweredOps.contains({defOp, phase}))
        continue;
      if (!opsSeen.insert({defOp, phase}).second) {
        defOp->emitOpError("is on a combinational loop");
        dumpWorklist();
        return failure();
      }
      opsWorklist.push_back(OpLowering(defOp, phase, *this));
      continue;
    }

    // At this point all operands are available and the op itself can be
    // lowered.
    LLVM_DEBUG(llvm::dbgs() << "  - Lowering " << opLowering.phase << " "
                            << *opLowering.op << "\n");
    if (failed(opLowering.lower())) {
      dumpWorklist();
      return failure();
    }
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

  // Handle memories.
  if (auto memOp = dyn_cast<MemoryOp>(result.getOwner())) {
    auto alloc = allocBuilder.create<AllocMemoryOp>(
        memOp.getLoc(), memOp.getType(), storageArg, memOp->getAttrs());
    allocatedStates.insert({result, alloc});
    return alloc;
  }

  // Create the allocation op.
  auto alloc = allocBuilder.create<AllocStateOp>(
      result.getLoc(), StateType::get(result.getType()), storageArg);
  allocatedStates.insert({result, alloc});

  // HACK: If the result comes from an instance op, add the instance and port
  // name as an attribute to the allocation. This will make it show up in the C
  // headers later. Get rid of this once we have proper debug dialect support.
  if (auto instOp = dyn_cast<InstanceOp>(result.getOwner()))
    alloc->setAttr(
        "name", builder.getStringAttr(
                    instOp.getInstanceName() + "/" +
                    instOp.getResultName(result.getResultNumber()).getValue()));

  // HACK: If the result comes from an op that has a "names" attribute, use that
  // as a name for the allocation. This should no longer be necessary once we
  // properly support the Debug dialect.
  if (isa<StateOp>(result.getOwner()))
    if (auto names = result.getOwner()->getAttrOfType<ArrayAttr>("names"))
      if (result.getResultNumber() < names.size())
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
  auto oldStorage = allocBuilder.create<AllocStateOp>(
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

/// Get the builder appropriate for the given phase.
OpBuilder &ModuleLowering::getBuilder(Phase phase) {
  switch (phase) {
  case Phase::Initial:
    return initialBuilder;
  case Phase::Old:
  case Phase::New:
    return builder;
  case Phase::Final:
    return finalBuilder;
  }
}

/// Get the lowered value, or emit a diagnostic and return null.
Value ModuleLowering::requireLoweredValue(Value value, Phase phase,
                                          Location useLoc) {
  if (auto lowered = loweredValues.lookup({value, phase}))
    return lowered;
  auto d = emitError(useLoc) << "value has not been lowered";
  d.attachNote(value.getLoc()) << "value defined here";
  return {};
}

//===----------------------------------------------------------------------===//
// Operation Lowering
//===----------------------------------------------------------------------===//

/// Create a new `scf.if` operation with the given builder, or reuse a previous
/// `scf.if` if the builder's insertion point is located right after it.
static scf::IfOp createOrReuseIf(OpBuilder &builder, Value condition,
                                 bool withElse) {
  scf::IfOp ifClockOp;
  if (auto ip = builder.getInsertionPoint(); ip != builder.getBlock()->begin())
    if (auto ifOp = dyn_cast<scf::IfOp>(*std::prev(ip)))
      if (ifOp.getCondition() == condition)
        return ifOp;
  return builder.create<scf::IfOp>(condition.getLoc(), condition, withElse);
}

/// This function is called from the lowering worklist in order to perform a
/// depth-first traversal of the surrounding module. These functions call
/// `lowerValue` to mark their operands as dependencies in the depth-first
/// traversal, and to map them to the lowered value in one go.
LogicalResult OpLowering::lower() {
  return TypeSwitch<Operation *, LogicalResult>(op)
      // Operations with special lowering.
      .Case<StateOp, MemoryOp, TapOp, InstanceOp, hw::OutputOp, seq::InitialOp>(
          [&](auto op) { return lower(op); })

      // Operations that should be skipped entirely and never land on the
      // worklist to be lowered.
      .Case<MemoryWritePortOp, MemoryReadPortOp>([&](auto op) {
        op.emitOpError() << "is handled by memory op and must be skipped";
        return success();
      })

      // Not yet supported.
      .Case<sim::DPICallOp>([&](auto op) {
        if (initial)
          return success();
        op.emitOpError() << "state lowering not yet implemented";
        return failure();
      })

      // All other ops are simply cloned into the lowered model.
      .Default([&](auto) { return lowerDefault(); });
}

/// Called for all operations for which there is no special lowering. Simply
/// clones the operation.
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

/// Lower a state to a corresponding storage allocation and write of the state's
/// new value to it. This function uses the `Old` phase to get the values at the
/// state input before the current update, and then uses them to compute the
/// `New` value.
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
  if (op.getLatency() > 1)
    return op.emitOpError("latencies > 1 not supported yet");

  // Check if we're inserting right after an if op for the same clock edge, in
  // which case we can reuse that op. Otherwise create the new if op.
  auto ifClockOp = createIfClockOp(op.getClock());
  if (!ifClockOp)
    return failure();
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
    auto ifResetOp = createOrReuseIf(module.builder, reset, true);
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
    auto ifEnableOp = createOrReuseIf(module.builder, enable, false);
    module.builder.setInsertionPoint(ifEnableOp.thenYield());
  }

  // Get the transfer function inputs. This potentially inserts read ops.
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

/// Lower a memory and its read and write ports to corresponding
/// `arc.memory_write` operations. Reads are also executed at this point and
/// stored in `loweredValues` for later operations to pick up.
LogicalResult OpLowering::lower(MemoryOp op) {
  if (phase != Phase::New)
    return op.emitOpError() << "cannot be lowered in the " << phase << " phase";

  // Collect all the reads and writes.
  SmallVector<MemoryReadPortOp> reads;
  SmallVector<MemoryWritePortOp> writes;

  for (auto *user : op->getUsers()) {
    if (auto read = dyn_cast<MemoryReadPortOp>(user)) {
      reads.push_back(read);
    } else if (auto write = dyn_cast<MemoryWritePortOp>(user)) {
      writes.push_back(write);
    } else {
      auto d = op.emitOpError()
               << "users must all be memory read or write port ops";
      d.attachNote(user->getLoc())
          << "but found " << user->getName() << " user here";
      return d;
    }
  }

  // Ensure all operands are lowered before we lower the memory itself.
  if (initial) {
    for (auto read : reads)
      lowerValue(read, Phase::Old);
    for (auto write : writes) {
      if (write.getClock())
        lowerValue(write.getClock(), Phase::New);
      for (auto input : write.getInputs())
        lowerValue(input, Phase::Old);
    }
    return success();
  }

  // Get the allocated storage for the memory.
  auto state = module.getAllocatedState(op->getResult(0));

  // Since we are going to write new values into storage, insert read ops that
  // keep the old values around for any later ops that still need them.
  for (auto read : reads) {
    auto oldValue = lowerValue(read, Phase::Old);
    if (!oldValue)
      return failure();
    module.loweredValues[{read, Phase::Old}] = oldValue;
  }

  // Lower the writes.
  for (auto write : writes) {
    if (!write.getClock())
      return write.emitOpError() << "must have a clock";
    if (write.getLatency() > 1)
      return write.emitOpError("latencies > 1 not supported yet");

    // Create the if op for the clock edge.
    auto ifClockOp = createIfClockOp(write.getClock());
    if (!ifClockOp)
      return failure();
    OpBuilder::InsertionGuard guard(module.builder);
    module.builder.setInsertionPoint(ifClockOp.thenYield());

    // Call the arc that computes the address, data, and enable.
    SmallVector<Value> inputs;
    for (auto input : write.getInputs()) {
      auto lowered = lowerValue(input, Phase::Old);
      if (!lowered)
        return failure();
      inputs.push_back(lowered);
    }
    auto callOp = module.builder.create<CallOp>(
        write.getLoc(), write.getArcResultTypes(), write.getArc(), inputs);

    // If the write has an enable, wrap the remaining logic in an if op.
    if (write.getEnable()) {
      auto ifEnableOp = createOrReuseIf(
          module.builder, callOp.getResult(write.getEnableIdx()), false);
      module.builder.setInsertionPoint(ifEnableOp.thenYield());
    }

    // If the write is masked, read the current
    // value in the memory and merge it with the updated value.
    auto address = callOp.getResult(write.getAddressIdx());
    auto data = callOp.getResult(write.getDataIdx());
    if (write.getMask()) {
      auto mask = callOp.getResult(write.getMaskIdx(write.getEnable()));
      auto maskInv = module.builder.createOrFold<comb::XorOp>(
          write.getLoc(), mask,
          module.builder.create<ConstantOp>(write.getLoc(), mask.getType(),
                                            -1));
      auto oldData =
          module.builder.create<MemoryReadOp>(write.getLoc(), state, address);
      auto oldMasked = module.builder.create<comb::AndOp>(
          write.getLoc(), maskInv, oldData, true);
      auto newMasked =
          module.builder.create<comb::AndOp>(write.getLoc(), mask, data, true);
      data = module.builder.create<comb::OrOp>(write.getLoc(), oldMasked,
                                               newMasked, true);
    }

    // Actually write to the memory.
    module.builder.create<MemoryWriteOp>(write.getLoc(), state, address,
                                         Value{}, data);
  }

  return success();
}

/// Lower a tap by allocating state storage for it and writing the current value
/// observed by the tap to it.
LogicalResult OpLowering::lower(TapOp op) {
  auto value = lowerValue(op.getValue(), phase);
  if (initial)
    return success();
  if (!value)
    return failure();

  auto &state = module.allocatedTaps[op];
  if (!state) {
    auto alloc = module.allocBuilder.create<AllocStateOp>(
        op.getLoc(), StateType::get(value.getType()), module.storageArg, true);
    alloc->setAttr("name", op.getNameAttr());
    state = alloc;
  }
  module.getBuilder(phase).create<StateWriteOp>(op.getLoc(), state, value,
                                                Value{});
  return success();
}

/// Lower an instance by allocating state storage for each of its inputs and
/// writing the current value into that storage. This makes instance inputs
/// behave like outputs of the top-level module.
LogicalResult OpLowering::lower(InstanceOp op) {
  // Get the current values flowing into the instance's inputs.
  SmallVector<Value> values;
  for (auto operand : op.getOperands())
    values.push_back(lowerValue(operand, Phase::New));
  if (initial)
    return success();
  if (llvm::is_contained(values, Value{}))
    return failure();

  // Then allocate storage for each instance input and assign the corresponding
  // value.
  for (auto [value, name] : llvm::zip(values, op.getArgNames())) {
    auto state = module.allocBuilder.create<AllocStateOp>(
        value.getLoc(), StateType::get(value.getType()), module.storageArg);
    state->setAttr("name", module.builder.getStringAttr(
                               op.getInstanceName() + "/" +
                               cast<StringAttr>(name).getValue()));
    module.builder.create<StateWriteOp>(value.getLoc(), state, value, Value{});
  }

  // HACK: Also ensure that storage has been allocated for all outputs.
  // Otherwise only the actually used instance outputs would be allocated, which
  // would make the optimization user-visible. Remove this once we use the debug
  // dialect.
  for (auto result : op.getResults())
    module.getAllocatedState(result);

  return success();
}

/// Lower the main module's outputs by allocating storage for each and then
/// writing the current value into that storage.
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
    auto state = module.allocBuilder.create<RootOutputOp>(
        value.getLoc(), StateType::get(value.getType()), cast<StringAttr>(name),
        module.storageArg);
    module.builder.create<StateWriteOp>(value.getLoc(), state, value, Value{});
  }
  return success();
}

LogicalResult OpLowering::lower(seq::InitialOp initialOp) {
  if (phase != Phase::Initial)
    return initialOp.emitOpError()
           << "cannot be lowered in the " << phase << " phase";

  // First get the initial value of all operands.
  SmallVector<Value> operands;
  for (auto operand : initialOp.getOperands())
    operands.push_back(lowerValue(operand, Phase::Initial));
  if (initial)
    return success();
  if (llvm::is_contained(operands, Value{}))
    return failure();

  // Expose the `seq.initial` operands as values for the block arguments.
  for (auto [arg, operand] :
       llvm::zip(initialOp.getBody().getArguments(), operands))
    module.loweredValues[{arg, Phase::Initial}] = operand;

  // Lower each op in the body.
  for (auto &op : initialOp.getOps()) {
    if (isa<seq::YieldOp>(op))
      continue;

    // Clone the operation.
    auto *clonedOp = module.initialBuilder.clone(op);
    auto result = clonedOp->walk([&](Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        if (clonedOp->isAncestor(operand.get().getParentBlock()->getParentOp()))
          continue;
        auto value = module.requireLoweredValue(operand.get(), Phase::Initial,
                                                op->getLoc());
        if (!value)
          return WalkResult::interrupt();
        operand.set(value);
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    // Keep track of the results.
    for (auto [result, lowered] :
         llvm::zip(op.getResults(), clonedOp->getResults()))
      module.loweredValues[{result, Phase::Initial}] = lowered;
  }

  // Expose the operands of `seq.yield` as results from the initial op.
  auto *terminator = initialOp.getBodyBlock()->getTerminator();
  for (auto [result, operand] :
       llvm::zip(initialOp.getResults(), terminator->getOperands())) {
    auto value = module.requireLoweredValue(operand, Phase::Initial,
                                            terminator->getLoc());
    if (!value)
      return failure();
    module.loweredValues[{result, Phase::Initial}] = value;
  }

  return success();
}

/// Create the operations necessary to detect a posedge on the given clock,
/// potentially reusing a previous posedge detection, and create an `scf.if`
/// operation for that posedge. This also tries to reuse an `scf.if` operation
/// immediately before the builder's insertion point if possible.
scf::IfOp OpLowering::createIfClockOp(Value clock) {
  auto &posedge = module.loweredPosedges[clock];
  if (!posedge) {
    auto loweredClock = lowerValue(clock, Phase::New);
    if (!loweredClock)
      return {};
    posedge = module.detectPosedge(loweredClock);
  }
  return createOrReuseIf(module.builder, posedge, false);
}

//===----------------------------------------------------------------------===//
// Value Lowering
//===----------------------------------------------------------------------===//

/// Lower a value being used by the current operation. This will mark the
/// defining operation as to be lowered first (through `addPending`) in most
/// cases. Some operations and values have special handling though. For example,
/// states and memory reads are immediately materialized as a new read op.
Value OpLowering::lowerValue(Value value, Phase phase) {
  // Handle module inputs. They read the same in all phases.
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (initial)
      return {};
    auto state = module.allocatedInputs[arg.getArgNumber()];
    return module.builder.create<StateReadOp>(arg.getLoc(), state);
  }

  // Check if the value has already been lowered.
  if (auto lowered = module.loweredValues.lookup({value, phase}))
    return lowered;

  // At this point the value is the result of an op. (Block arguments are
  // handled above.)
  auto result = cast<OpResult>(value);
  auto *op = result.getOwner();

  // Special handling for some ops.
  if (auto instOp = dyn_cast<InstanceOp>(op))
    return lowerValue(instOp, result, phase);
  if (auto stateOp = dyn_cast<StateOp>(op))
    return lowerValue(stateOp, result, phase);
  if (auto readOp = dyn_cast<MemoryReadPortOp>(op))
    return lowerValue(readOp, result, phase);
  if (auto castOp = dyn_cast<seq::FromImmutableOp>(op))
    return lowerValue(castOp, result, phase);

  // Otherwise we mark the defining operation as to be lowered first. This will
  // cause the lookup in `loweredValues` above to return a value the next time
  // (i.e. when initial is false).
  if (initial) {
    addPending(op, phase);
    return {};
  }
  auto d = this->op->emitOpError() << "operand value has not been lowered";
  d.attachNote(result.getLoc()) << "value defined here";
  return {};
}

/// Handle instance outputs. They behave essentially like a top-level module
/// input, and read the same in all phases.
Value OpLowering::lowerValue(InstanceOp op, OpResult result, Phase phase) {
  if (initial)
    return {};
  auto state = module.getAllocatedState(result);
  return module.builder.create<StateReadOp>(result.getLoc(), state);
}

/// Handle uses of a state. This creates a `arc.state_read` op to read from the
/// state's storage. If the new value after all updates is requested, marks the
/// state as to be lowered first (which will perform the writes). If the old
/// value is requested, asserts that no new values have been written.
Value OpLowering::lowerValue(StateOp op, OpResult result, Phase phase) {
  if (initial) {
    // Ensure that new value is written before we attempt to read it.
    if (phase == Phase::New)
      addPending(op, Phase::New);
    return {};
  }

  if (phase == Phase::Old) {
    // If we want to read the old value, no writes must have been lowered yet.
    assert(!module.loweredOps.contains({op, Phase::New}) &&
           "need old value but new value already written");
  } else if (phase != Phase::New) {
    op.emitOpError() << "result cannot be used in " << phase << " phase\n";
    return {};
  }

  auto state = module.getAllocatedState(result);
  return module.builder.create<StateReadOp>(result.getLoc(), state);
}

/// Handle uses of a memory read operation. This creates an `arc.memory_read` op
/// to read from the memory's storage. Similar to the `StateOp` handling
/// otherwise.
Value OpLowering::lowerValue(MemoryReadPortOp op, OpResult result,
                             Phase phase) {
  auto memOp = op.getMemory().getDefiningOp<MemoryOp>();
  if (!memOp) {
    if (!initial)
      op->emitOpError() << "memory must be defined locally";
    return {};
  }

  auto address = lowerValue(op.getAddress(), phase);
  if (initial) {
    // Ensure that all new values are written before we attempt to read them.
    if (phase == Phase::New)
      addPending(memOp.getOperation(), Phase::New);
    return {};
  }
  if (!address)
    return {};

  if (phase == Phase::Old) {
    // If we want to read the old value, no writes must have been lowered yet.
    assert(!module.loweredOps.contains({memOp, Phase::New}) &&
           "need old memory value but new value already written");
  } else if (phase != Phase::New) {
    op.emitOpError() << "result cannot be used in " << phase << " phase\n";
    return {};
  }

  auto state = module.getAllocatedState(memOp->getResult(0));
  return module.builder.create<MemoryReadOp>(result.getLoc(), state, address);
}

/// Handle uses of values computed during the initial phase. This ensures that
/// the interesting value is stored into storage during the initial phase, and
/// then reads it back using an `arc.state_read` op.
Value OpLowering::lowerValue(seq::FromImmutableOp op, OpResult result,
                             Phase phase) {
  // Ensure the input to the cast is lowered first.
  auto value = lowerValue(op.getInput(), Phase::Initial);
  if (initial || !value)
    return {};

  // If necessary, allocate storage for the computed value and store it in the
  // initial phase.
  auto &state = module.allocatedInitials[value];
  if (!state) {
    state = module.allocBuilder.create<AllocStateOp>(
        value.getLoc(), StateType::get(value.getType()), module.storageArg);
    OpBuilder::InsertionGuard guard(module.initialBuilder);
    module.initialBuilder.setInsertionPointAfterValue(value);
    module.initialBuilder.create<StateWriteOp>(value.getLoc(), state, value,
                                               Value{});
  }

  // Read back the value computed during the initial phase.
  return module.builder.create<StateReadOp>(state.getLoc(), state);
}

/// Mark a value as to be lowered before the current op.
void OpLowering::addPending(Value value, Phase phase) {
  auto *defOp = value.getDefiningOp();
  assert(defOp && "block args should never be marked as a dependency");
  addPending(defOp, phase);
}

/// Mark an operation as to be lowered before the current op. This adds that
/// operation to the `pending` list if the operation has not yet been lowered.
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
struct LowerStatePass : public arc::impl::LowerStatePassBase<LowerStatePass> {
  using LowerStatePassBase::LowerStatePassBase;
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
