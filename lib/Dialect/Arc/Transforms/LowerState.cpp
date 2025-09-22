//===- LowerState.cpp -----------------------------------------------------===//
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
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
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
  LogicalResult lower(sim::DPICallOp op);
  LogicalResult
  lowerStateful(Value clock, Value enable, Value reset, ValueRange inputs,
                ResultRange results,
                llvm::function_ref<ValueRange(ValueRange)> createMapping);
  LogicalResult lower(MemoryOp op);
  LogicalResult lower(TapOp op);
  LogicalResult lower(InstanceOp op);
  LogicalResult lower(hw::OutputOp op);
  LogicalResult lower(seq::InitialOp op);
  LogicalResult lower(llhd::FinalOp op);

  scf::IfOp createIfClockOp(Value clock);

  // Value Lowering. These functions are called from the `lower()` functions
  // above. They handle values used by the `op`. This can generate reads from
  // state and memory storage on-the-fly, or mark other ops as dependencies to
  // be lowered first.
  Value lowerValue(Value value, Phase phase);
  Value lowerValue(InstanceOp op, OpResult result, Phase phase);
  Value lowerValue(StateOp op, OpResult result, Phase phase);
  Value lowerValue(sim::DPICallOp op, OpResult result, Phase phase);
  Value lowerValue(MemoryReadPortOp op, OpResult result, Phase phase);
  Value lowerValue(seq::InitialOp op, OpResult result, Phase phase);
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
  /// The allocated storage for instance inputs and top module outputs.
  DenseMap<OpOperand *, Value> allocatedOutputs;
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
  /// previous if ops for the same reset value.
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
  auto modelOp =
      ModelOp::create(builder, moduleOp.getLoc(), moduleOp.getModuleNameAttr(),
                      TypeAttr::get(moduleOp.getModuleType()),
                      FlatSymbolRefAttr{}, FlatSymbolRefAttr{});
  auto &modelBlock = modelOp.getBody().emplaceBlock();
  storageArg = modelBlock.addArgument(
      StorageType::get(builder.getContext(), {}), modelOp.getLoc());
  builder.setInsertionPointToStart(&modelBlock);

  // Create the `arc.initial` op to contain the ops for the initialization
  // phase.
  auto initialOp = InitialOp::create(builder, moduleOp.getLoc());
  initialBuilder.setInsertionPointToStart(&initialOp.getBody().emplaceBlock());

  // Create the `arc.final` op to contain the ops for the finalization phase.
  auto finalOp = FinalOp::create(builder, moduleOp.getLoc());
  finalBuilder.setInsertionPointToStart(&finalOp.getBody().emplaceBlock());

  // Position the alloc builder such that allocation ops get inserted above the
  // initial op.
  allocBuilder.setInsertionPoint(initialOp);

  // Allocate storage for the inputs.
  for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
    auto name = moduleOp.getArgName(arg.getArgNumber());
    auto state =
        RootInputOp::create(allocBuilder, arg.getLoc(),
                            StateType::get(arg.getType()), name, storageArg);
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

  // Pick in which phases the given operation has to perform some work.
  SmallVector<Phase, 2> phases = {Phase::New};
  if (isa<seq::InitialOp>(op))
    phases = {Phase::Initial};
  if (isa<llhd::FinalOp>(op))
    phases = {Phase::Final};
  if (isa<StateOp>(op))
    phases = {Phase::Initial, Phase::New};

  for (auto phase : phases) {
    if (loweredOps.contains({op, phase}))
      return success();
    opsWorklist.push_back(OpLowering(op, phase, *this));
    opsSeen.insert({op, phase});
  }

  auto dumpWorklist = [&] {
    for (auto &opLowering : llvm::reverse(opsWorklist))
      opLowering.op->emitRemark()
          << "computing " << opLowering.phase << " phase here";
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
    auto alloc =
        AllocMemoryOp::create(allocBuilder, memOp.getLoc(), memOp.getType(),
                              storageArg, memOp->getAttrs());
    allocatedStates.insert({result, alloc});
    return alloc;
  }

  // Create the allocation op.
  auto alloc =
      AllocStateOp::create(allocBuilder, result.getLoc(),
                           StateType::get(result.getType()), storageArg);
  allocatedStates.insert({result, alloc});

  // HACK: If the result comes from an instance op, add the instance and port
  // name as an attribute to the allocation. This will make it show up in the C
  // headers later. Get rid of this once we have proper debug dialect support.
  if (auto instOp = dyn_cast<InstanceOp>(result.getOwner()))
    alloc->setAttr(
        "name", builder.getStringAttr(
                    instOp.getInstanceName() + "/" +
                    instOp.getOutputName(result.getResultNumber()).getValue()));

  // HACK: If the result comes from an op that has a "names" attribute, use that
  // as a name for the allocation. This should no longer be necessary once we
  // properly support the Debug dialect.
  if (isa<StateOp, sim::DPICallOp>(result.getOwner()))
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
    clock = seq::FromClockOp::create(builder, loc, clock);

  // Allocate storage to store the previous clock value.
  auto oldStorage = AllocStateOp::create(
      allocBuilder, loc, StateType::get(builder.getI1Type()), storageArg);

  // Read the old clock value from storage and write the new clock value to
  // storage.
  auto oldClock = StateReadOp::create(builder, loc, oldStorage);
  StateWriteOp::create(builder, loc, oldStorage, clock, Value{});

  // Detect a rising edge.
  auto edge = comb::XorOp::create(builder, loc, oldClock, clock);
  return comb::AndOp::create(builder, loc, edge, clock);
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
  auto d = emitError(value.getLoc()) << "value has not been lowered";
  d.attachNote(useLoc) << "value used here";
  return {};
}

//===----------------------------------------------------------------------===//
// Operation Lowering
//===----------------------------------------------------------------------===//

/// Create a new `scf.if` operation with the given builder, or reuse a previous
/// `scf.if` if the builder's insertion point is located right after it.
static scf::IfOp createOrReuseIf(OpBuilder &builder, Value condition,
                                 bool withElse) {
  if (auto ip = builder.getInsertionPoint(); ip != builder.getBlock()->begin())
    if (auto ifOp = dyn_cast<scf::IfOp>(*std::prev(ip)))
      if (ifOp.getCondition() == condition)
        return ifOp;
  return scf::IfOp::create(builder, condition.getLoc(), condition, withElse);
}

/// This function is called from the lowering worklist in order to perform a
/// depth-first traversal of the surrounding module. These functions call
/// `lowerValue` to mark their operands as dependencies in the depth-first
/// traversal, and to map them to the lowered value in one go.
LogicalResult OpLowering::lower() {
  return TypeSwitch<Operation *, LogicalResult>(op)
      // Operations with special lowering.
      .Case<StateOp, sim::DPICallOp, MemoryOp, TapOp, InstanceOp, hw::OutputOp,
            seq::InitialOp, llhd::FinalOp>([&](auto op) { return lower(op); })

      // Operations that should be skipped entirely and never land on the
      // worklist to be lowered.
      .Case<MemoryWritePortOp, MemoryReadPortOp>([&](auto op) {
        assert(false && "ports must be lowered by memory op");
        return failure();
      })

      // All other ops are simply cloned into the lowered model.
      .Default([&](auto) { return lowerDefault(); });
}

/// Called for all operations for which there is no special lowering. Simply
/// clones the operation.
LogicalResult OpLowering::lowerDefault() {
  // Make sure that all operand values are lowered first.
  IRMapping mapping;
  auto anyFailed = false;
  op->walk([&](Operation *nestedOp) {
    for (auto operand : nestedOp->getOperands()) {
      if (op->isAncestor(operand.getParentBlock()->getParentOp()))
        continue;
      auto lowered = lowerValue(operand, phase);
      if (!lowered)
        anyFailed = true;
      mapping.map(operand, lowered);
    }
  });
  if (initial)
    return success();
  if (anyFailed)
    return failure();

  // Clone the operation.
  auto *clonedOp = module.getBuilder(phase).clone(*op, mapping);

  // Keep track of the results.
  for (auto [oldResult, newResult] :
       llvm::zip(op->getResults(), clonedOp->getResults()))
    module.loweredValues[{oldResult, phase}] = newResult;

  return success();
}

/// Lower a state to a corresponding storage allocation and `write` of the
/// state's new value to it. This function uses the `Old` phase to get the
/// values at the state input before the current update, and then uses them to
/// compute the `New` value.
LogicalResult OpLowering::lower(StateOp op) {
  // Handle initialization.
  if (phase == Phase::Initial) {
    // Ensure the initial values of the register have been lowered before.
    if (initial) {
      for (auto initial : op.getInitials())
        lowerValue(initial, Phase::Initial);
      return success();
    }

    // Write the initial values to the allocated storage in the initial block.
    if (op.getInitials().empty())
      return success();
    for (auto [initial, result] :
         llvm::zip(op.getInitials(), op.getResults())) {
      auto value = lowerValue(initial, Phase::Initial);
      if (!value)
        return failure();
      auto state = module.getAllocatedState(result);
      if (!state)
        return failure();
      StateWriteOp::create(module.initialBuilder, value.getLoc(), state, value,
                           Value{});
    }
    return success();
  }

  assert(phase == Phase::New);

  if (!initial) {
    if (!op.getClock())
      return op.emitOpError() << "must have a clock";
    if (op.getLatency() > 1)
      return op.emitOpError("latencies > 1 not supported yet");
  }

  return lowerStateful(op.getClock(), op.getEnable(), op.getReset(),
                       op.getInputs(), op.getResults(), [&](ValueRange inputs) {
                         return CallOp::create(module.builder, op.getLoc(),
                                               op.getResultTypes(), op.getArc(),
                                               inputs)
                             .getResults();
                       });
}

/// Lower a DPI call to a corresponding storage allocation and write of the
/// state's new value to it. This function uses the `Old` phase to get the
/// values at the state input before the current update, and then uses them to
/// compute the `New` value.
LogicalResult OpLowering::lower(sim::DPICallOp op) {
  // Handle unclocked DPI calls.
  if (!op.getClock()) {
    // Make sure that all operands have been lowered.
    SmallVector<Value> inputs;
    for (auto operand : op.getInputs())
      inputs.push_back(lowerValue(operand, phase));
    if (initial)
      return success();
    if (llvm::is_contained(inputs, Value{}))
      return failure();
    if (op.getEnable())
      return op.emitOpError() << "without clock cannot have an enable";

    // Lower the op to a regular function call.
    auto callOp =
        func::CallOp::create(module.getBuilder(phase), op.getLoc(),
                             op.getCalleeAttr(), op.getResultTypes(), inputs);
    for (auto [oldResult, newResult] :
         llvm::zip(op.getResults(), callOp.getResults()))
      module.loweredValues[{oldResult, phase}] = newResult;
    return success();
  }

  assert(phase == Phase::New);

  return lowerStateful(op.getClock(), op.getEnable(), /*reset=*/{},
                       op.getInputs(), op.getResults(), [&](ValueRange inputs) {
                         return func::CallOp::create(
                                    module.builder, op.getLoc(),
                                    op.getCalleeAttr(), op.getResultTypes(),
                                    inputs)
                             .getResults();
                       });
}

/// Lower a state to a corresponding storage allocation and `write` of the
/// state's new value to it. This function uses the `Old` phase to get the
/// values at the state input before the current update, and then uses them to
/// compute the `New` value.
LogicalResult OpLowering::lowerStateful(
    Value clock, Value enable, Value reset, ValueRange inputs,
    ResultRange results,
    llvm::function_ref<ValueRange(ValueRange)> createMapping) {
  // Ensure all operands are lowered before we lower the op itself. State ops
  // are special in that they require the "old" value of their inputs and
  // enable, in order to compute the updated "new" value. The clock needs to be
  // the "new" value though, such that other states can act as a clock source.
  if (initial) {
    lowerValue(clock, Phase::New);
    if (enable)
      lowerValue(enable, Phase::Old);
    if (reset)
      lowerValue(reset, Phase::Old);
    for (auto input : inputs)
      lowerValue(input, Phase::Old);
    return success();
  }

  // Check if we're inserting right after an `if` op for the same clock edge, in
  // which case we can reuse that op. Otherwise, create the new `if` op.
  auto ifClockOp = createIfClockOp(clock);
  if (!ifClockOp)
    return failure();
  OpBuilder::InsertionGuard guard(module.builder);
  module.builder.setInsertionPoint(ifClockOp.thenYield());

  // Make sure we have the state storage available such that we can read and
  // write from and to them.
  SmallVector<Value> states;
  for (auto result : results) {
    auto state = module.getAllocatedState(result);
    if (!state)
      return failure();
    states.push_back(state);
  }

  // Handle the reset.
  if (reset) {
    // Check if we can reuse a previous reset value.
    auto &[unloweredReset, loweredReset] = module.prevReset;
    if (unloweredReset != reset ||
        loweredReset.getParentBlock() != module.builder.getBlock()) {
      unloweredReset = reset;
      loweredReset = lowerValue(reset, Phase::Old);
      if (!loweredReset)
        return failure();
    }

    // Check if we're inserting right after an if op for the same reset, in
    // which case we can reuse that op. Otherwise create the new if op.
    auto ifResetOp = createOrReuseIf(module.builder, loweredReset, true);
    module.builder.setInsertionPoint(ifResetOp.thenYield());

    // Generate the zero value writes.
    for (auto state : states) {
      auto type = cast<StateType>(state.getType()).getType();
      Value value = ConstantOp::create(
          module.builder, loweredReset.getLoc(),
          module.builder.getIntegerType(hw::getBitWidth(type)), 0);
      if (value.getType() != type)
        value = BitcastOp::create(module.builder, loweredReset.getLoc(), type,
                                  value);
      StateWriteOp::create(module.builder, loweredReset.getLoc(), state, value,
                           Value{});
    }
    module.builder.setInsertionPoint(ifResetOp.elseYield());
  }

  // Handle the enable.
  if (enable) {
    // Check if we can reuse a previous enable value.
    auto &[unloweredEnable, loweredEnable] = module.prevEnable;
    if (unloweredEnable != enable ||
        loweredEnable.getParentBlock() != module.builder.getBlock()) {
      unloweredEnable = enable;
      loweredEnable = lowerValue(enable, Phase::Old);
      if (!loweredEnable)
        return failure();
    }

    // Check if we're inserting right after an if op for the same enable, in
    // which case we can reuse that op. Otherwise create the new if op.
    auto ifEnableOp = createOrReuseIf(module.builder, loweredEnable, false);
    module.builder.setInsertionPoint(ifEnableOp.thenYield());
  }

  // Get the transfer function inputs. This potentially inserts read ops.
  SmallVector<Value> loweredInputs;
  for (auto input : inputs) {
    auto lowered = lowerValue(input, Phase::Old);
    if (!lowered)
      return failure();
    loweredInputs.push_back(lowered);
  }

  // Compute the transfer function and write its results to the state's storage.
  auto loweredResults = createMapping(loweredInputs);
  for (auto [state, value] : llvm::zip(states, loweredResults))
    StateWriteOp::create(module.builder, value.getLoc(), state, value, Value{});

  // Since we just wrote the new state value to storage, insert read ops just
  // before the if op that keep the old value around for any later ops that
  // still need it.
  module.builder.setInsertionPoint(ifClockOp);
  for (auto [state, result] : llvm::zip(states, results)) {
    auto oldValue = StateReadOp::create(module.builder, result.getLoc(), state);
    module.loweredValues[{result, Phase::Old}] = oldValue;
  }

  return success();
}

/// Lower a memory and its read and write ports to corresponding
/// `arc.memory_write` operations. Reads are also executed at this point and
/// stored in `loweredValues` for later operations to pick up.
LogicalResult OpLowering::lower(MemoryOp op) {
  assert(phase == Phase::New);

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
    auto callOp =
        CallOp::create(module.builder, write.getLoc(),
                       write.getArcResultTypes(), write.getArc(), inputs);

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
          ConstantOp::create(module.builder, write.getLoc(), mask.getType(),
                             -1),
          true);
      auto oldData =
          MemoryReadOp::create(module.builder, write.getLoc(), state, address);
      auto oldMasked = comb::AndOp::create(module.builder, write.getLoc(),
                                           maskInv, oldData, true);
      auto newMasked =
          comb::AndOp::create(module.builder, write.getLoc(), mask, data, true);
      data = comb::OrOp::create(module.builder, write.getLoc(), oldMasked,
                                newMasked, true);
    }

    // Actually write to the memory.
    MemoryWriteOp::create(module.builder, write.getLoc(), state, address,
                          Value{}, data);
  }

  return success();
}

/// Lower a tap by allocating state storage for it and writing the current value
/// observed by the tap to it.
LogicalResult OpLowering::lower(TapOp op) {
  assert(phase == Phase::New);

  auto value = lowerValue(op.getValue(), phase);
  if (initial)
    return success();
  if (!value)
    return failure();

  auto &state = module.allocatedTaps[op];
  if (!state) {
    auto alloc = AllocStateOp::create(module.allocBuilder, op.getLoc(),
                                      StateType::get(value.getType()),
                                      module.storageArg, true);
    alloc->setAttr("names", op.getNamesAttr());
    state = alloc;
  }
  StateWriteOp::create(module.builder, op.getLoc(), state, value, Value{});
  return success();
}

/// Lower an instance by allocating state storage for each of its inputs and
/// writing the current value into that storage. This makes instance inputs
/// behave like outputs of the top-level module.
LogicalResult OpLowering::lower(InstanceOp op) {
  assert(phase == Phase::New);

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
    auto state = AllocStateOp::create(module.allocBuilder, value.getLoc(),
                                      StateType::get(value.getType()),
                                      module.storageArg);
    state->setAttr("name", module.builder.getStringAttr(
                               op.getInstanceName() + "/" +
                               cast<StringAttr>(name).getValue()));
    StateWriteOp::create(module.builder, value.getLoc(), state, value, Value{});
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
  assert(phase == Phase::New);

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
    auto state = RootOutputOp::create(
        module.allocBuilder, value.getLoc(), StateType::get(value.getType()),
        cast<StringAttr>(name), module.storageArg);
    StateWriteOp::create(module.builder, value.getLoc(), state, value, Value{});
  }
  return success();
}

/// Lower `seq.initial` ops by inlining them into the `arc.initial` op.
LogicalResult OpLowering::lower(seq::InitialOp op) {
  assert(phase == Phase::Initial);

  // First get the initial value of all operands.
  SmallVector<Value> operands;
  for (auto operand : op.getOperands())
    operands.push_back(lowerValue(operand, Phase::Initial));
  if (initial)
    return success();
  if (llvm::is_contained(operands, Value{}))
    return failure();

  // Expose the `seq.initial` operands as values for the block arguments.
  for (auto [arg, operand] : llvm::zip(op.getBody().getArguments(), operands))
    module.loweredValues[{arg, Phase::Initial}] = operand;

  // Lower each op in the body.
  for (auto &bodyOp : op.getOps()) {
    if (isa<seq::YieldOp>(bodyOp))
      continue;

    // Clone the operation.
    auto *clonedOp = module.initialBuilder.clone(bodyOp);
    auto result = clonedOp->walk([&](Operation *nestedClonedOp) {
      for (auto &operand : nestedClonedOp->getOpOperands()) {
        if (clonedOp->isAncestor(operand.get().getParentBlock()->getParentOp()))
          continue;
        auto value = module.requireLoweredValue(operand.get(), Phase::Initial,
                                                nestedClonedOp->getLoc());
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
         llvm::zip(bodyOp.getResults(), clonedOp->getResults()))
      module.loweredValues[{result, Phase::Initial}] = lowered;
  }

  // Expose the operands of `seq.yield` as results from the initial op.
  auto *terminator = op.getBodyBlock()->getTerminator();
  for (auto [result, operand] :
       llvm::zip(op.getResults(), terminator->getOperands())) {
    auto value = module.requireLoweredValue(operand, Phase::Initial,
                                            terminator->getLoc());
    if (!value)
      return failure();
    module.loweredValues[{result, Phase::Initial}] = value;
  }

  return success();
}

/// Lower `llhd.final` ops into `scf.execute_region` ops in the `arc.final` op.
LogicalResult OpLowering::lower(llhd::FinalOp op) {
  assert(phase == Phase::Final);

  // Determine the uses of values defined outside the op.
  SmallVector<Value> externalOperands;
  op.walk([&](Operation *nestedOp) {
    for (auto value : nestedOp->getOperands())
      if (!op->isAncestor(value.getParentBlock()->getParentOp()))
        externalOperands.push_back(value);
  });

  // Make sure that all uses of external values are lowered first.
  IRMapping mapping;
  for (auto operand : externalOperands) {
    auto lowered = lowerValue(operand, Phase::Final);
    if (!initial && !lowered)
      return failure();
    mapping.map(operand, lowered);
  }
  if (initial)
    return success();

  // Handle the simple case where the final op contains only one block, which we
  // can inline directly.
  if (op.getBody().hasOneBlock()) {
    for (auto &bodyOp : op.getBody().front().without_terminator())
      module.finalBuilder.clone(bodyOp, mapping);
    return success();
  }

  // Create a new `scf.execute_region` op and clone the entire `llhd.final` body
  // region into it. Replace `llhd.halt` ops with `scf.yield`.
  auto executeOp = scf::ExecuteRegionOp::create(module.finalBuilder,
                                                op.getLoc(), TypeRange{});
  module.finalBuilder.cloneRegionBefore(op.getBody(), executeOp.getRegion(),
                                        executeOp.getRegion().begin(), mapping);
  executeOp.walk([&](llhd::HaltOp op) {
    auto builder = OpBuilder(op);
    scf::YieldOp::create(builder, op.getLoc());
    op.erase();
  });

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
    return StateReadOp::create(module.getBuilder(phase), arg.getLoc(), state);
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
  if (auto dpiOp = dyn_cast<sim::DPICallOp>(op); dpiOp && dpiOp.getClock())
    return lowerValue(dpiOp, result, phase);
  if (auto readOp = dyn_cast<MemoryReadPortOp>(op))
    return lowerValue(readOp, result, phase);
  if (auto initialOp = dyn_cast<seq::InitialOp>(op))
    return lowerValue(initialOp, result, phase);
  if (auto castOp = dyn_cast<seq::FromImmutableOp>(op))
    return lowerValue(castOp, result, phase);

  // Otherwise we mark the defining operation as to be lowered first. This will
  // cause the lookup in `loweredValues` above to return a value the next time
  // (i.e. when initial is false).
  if (initial) {
    addPending(op, phase);
    return {};
  }
  emitError(result.getLoc()) << "value has not been lowered";
  return {};
}

/// Handle instance outputs. They behave essentially like a top-level module
/// input, and read the same in all phases.
Value OpLowering::lowerValue(InstanceOp op, OpResult result, Phase phase) {
  if (initial)
    return {};
  auto state = module.getAllocatedState(result);
  return StateReadOp::create(module.getBuilder(phase), result.getLoc(), state);
}

/// Handle uses of a state. This creates an `arc.state_read` op to read from the
/// state's storage. If the new value after all updates is requested, marks the
/// state as to be lowered first (which will perform the writes). If the old
/// value is requested, asserts that no new values have been written.
Value OpLowering::lowerValue(StateOp op, OpResult result, Phase phase) {
  if (initial) {
    // Ensure that the new or initial value has been written by the lowering of
    // the state op before we attempt to read it.
    if (phase == Phase::New || phase == Phase::Initial)
      addPending(op, phase);
    return {};
  }

  // If we want to read the old value, no writes must have been lowered yet.
  if (phase == Phase::Old)
    assert(!module.loweredOps.contains({op, Phase::New}) &&
           "need old value but new value already written");

  auto state = module.getAllocatedState(result);
  return StateReadOp::create(module.getBuilder(phase), result.getLoc(), state);
}

/// Handle uses of a DPI call. This creates an `arc.state_read` op to read from
/// the state's storage. If the new value after all updates is requested, marks
/// the state as to be lowered first (which will perform the writes). If the old
/// value is requested, asserts that no new values have been written.
Value OpLowering::lowerValue(sim::DPICallOp op, OpResult result, Phase phase) {
  if (initial) {
    // Ensure that the new or initial value has been written by the lowering of
    // the state op before we attempt to read it.
    if (phase == Phase::New || phase == Phase::Initial)
      addPending(op, phase);
    return {};
  }

  // If we want to read the old value, no writes must have been lowered yet.
  if (phase == Phase::Old)
    assert(!module.loweredOps.contains({op, Phase::New}) &&
           "need old value but new value already written");

  auto state = module.getAllocatedState(result);
  return StateReadOp::create(module.getBuilder(phase), result.getLoc(), state);
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
  } else {
    assert(phase == Phase::New);
  }

  auto state = module.getAllocatedState(memOp->getResult(0));
  return MemoryReadOp::create(module.getBuilder(phase), result.getLoc(), state,
                              address);
}

/// Handle uses of `seq.initial` values computed during the initial phase. This
/// ensures that the interesting value is stored into storage during the initial
/// phase, and then reads it back using an `arc.state_read` op.
Value OpLowering::lowerValue(seq::InitialOp op, OpResult result, Phase phase) {
  // Ensure the op has been lowered first.
  if (initial) {
    addPending(op, Phase::Initial);
    return {};
  }
  auto value = module.loweredValues.lookup({result, Phase::Initial});
  if (!value) {
    emitError(result.getLoc()) << "value has not been lowered";
    return {};
  }

  // If we are using the value of `seq.initial` in the initial phase directly,
  // there is no need to write it so any temporary storage.
  if (phase == Phase::Initial)
    return value;

  // If necessary, allocate storage for the computed value and store it in the
  // initial phase.
  auto &state = module.allocatedInitials[result];
  if (!state) {
    state = AllocStateOp::create(module.allocBuilder, value.getLoc(),
                                 StateType::get(value.getType()),
                                 module.storageArg);
    OpBuilder::InsertionGuard guard(module.initialBuilder);
    module.initialBuilder.setInsertionPointAfterValue(value);
    StateWriteOp::create(module.initialBuilder, value.getLoc(), state, value,
                         Value{});
  }

  // Read back the value computed during the initial phase.
  return StateReadOp::create(module.getBuilder(phase), state.getLoc(), state);
}

/// The `seq.from_immutable` cast is just a passthrough.
Value OpLowering::lowerValue(seq::FromImmutableOp op, OpResult result,
                             Phase phase) {
  return lowerValue(op.getInput(), phase);
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

  SymbolTable symbolTable(op);
  for (auto extModuleOp :
       llvm::make_early_inc_range(op.getOps<HWModuleExternOp>())) {
    // Make sure that we're not leaving behind a dangling reference to this
    // module
    auto uses = symbolTable.getSymbolUses(extModuleOp, op);
    if (!uses->empty()) {
      extModuleOp->emitError("Failed to remove external module because it is "
                             "still referenced/instantiated");
      return signalPassFailure();
    }
    extModuleOp.erase();
  }
}
