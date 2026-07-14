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
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  LogicalResult lower(CoroutineInstanceOp op);
  LogicalResult lower(hw::TriggeredOp op);
  LogicalResult lower(hw::OutputOp op);
  LogicalResult lower(seq::InitialOp op);
  LogicalResult lower(llhd::FinalOp op);
  LogicalResult lower(llhd::CurrentTimeOp op);
  LogicalResult lower(llhd::SignalOp op);
  LogicalResult lower(llhd::DriveOp op);
  LogicalResult lower(sim::ClockedTerminateOp op);

  scf::IfOp createIfClockOp(Value clock);

  // Value Lowering. These functions are called from the `lower()` functions
  // above. They handle values used by the `op`. This can generate reads from
  // state and memory storage on-the-fly, or mark other ops as dependencies to
  // be lowered first.
  Value lowerValue(Value value, Phase phase);
  Value lowerValue(InstanceOp op, OpResult result, Phase phase);
  Value lowerValue(CoroutineInstanceOp op, OpResult result, Phase phase);
  Value lowerValue(StateOp op, OpResult result, Phase phase);
  Value lowerValue(sim::DPICallOp op, OpResult result, Phase phase);
  Value lowerValue(MemoryReadPortOp op, OpResult result, Phase phase);
  Value lowerValue(seq::InitialOp op, OpResult result, Phase phase);
  Value lowerValue(seq::FromImmutableOp op, OpResult result, Phase phase);
  Value lowerValue(llhd::ProbeOp op, OpResult result, Phase phase);

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
  /// The builder for `Phase::Old` values: a pinned section at the top of the
  /// eval body, executed before any coroutine instance or drive can mutate
  /// signal storage. All "old" reads and the pure cones over them thus form a
  /// consistent pre-eval snapshot (IEEE 1800 preponed-style sampling for
  /// clocked state elements). Without this, an old read emitted after a
  /// coroutine call observes mid-eval process writes while cached reads from
  /// earlier positions observe pre-eval values: registers then commit a mix
  /// of pre- and post-process operands (the declaration-order-dependent
  /// stale-select NBA decode).
  OpBuilder oldBuilder;

  /// The storage value that can be used for `arc.alloc_state` and friends.
  Value storageArg;

  /// The symbol table of the enclosing top-level module. Used to resolve
  /// coroutine callees without walking the entire IR.
  SymbolTable &symbolTable;

  /// A worklist of pending op lowerings.
  SmallVector<OpLowering> opsWorklist;
  /// The set of ops currently in the worklist. Used to detect cycles.
  SmallDenseSet<std::pair<Operation *, Phase>> opsSeen;
  /// The ops that have already been lowered.
  DenseSet<std::pair<Operation *, Phase>> loweredOps;
  /// The values that have already been lowered.
  DenseMap<std::pair<Value, Phase>, Value> loweredValues;
  /// Module-level values that flow out of signal initializer expressions
  /// (`llhd.sig` init operands and everything derived from them at module
  /// level). Side-effecting module-level ops consuming these are one-time
  /// initialization actions (object memset, typeinfo/property-initializer
  /// stores emitted by class `new` at module scope) and must lower in the
  /// initial phase: lowering them per-eval re-executes the initializer
  /// chain against a fresh allocation (and leaks it every eval) while the
  /// signal keeps the pointer from its own, separate initial-phase clone.
  DenseSet<Value> sigInitClosure;

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

  ModuleLowering(HWModuleOp moduleOp, SymbolTable &symbolTable)
      : moduleOp(moduleOp), builder(moduleOp), allocBuilder(moduleOp),
        initialBuilder(moduleOp), finalBuilder(moduleOp), oldBuilder(moduleOp),
        symbolTable(symbolTable) {}
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
                      TypeAttr::get(moduleOp.getModuleType()), IntegerAttr{},
                      FlatSymbolRefAttr{}, FlatSymbolRefAttr{}, ArrayAttr{});
  auto &modelBlock = modelOp.getBody().emplaceBlock();
  storageArg = modelBlock.addArgument(StorageType::get(builder.getContext()),
                                      modelOp.getLoc());
  builder.setInsertionPointToStart(&modelBlock);

  // Reset the next wakeup slot to `UINT64_MAX` ("no wakeup pending") at the
  // start of every eval. Process suspension code lowers the value to the
  // earliest scheduled wakeup over the course of the evaluation.
  auto noWakeup = hw::ConstantOp::create(builder, moduleOp.getLoc(),
                                         builder.getI64Type(), -1);
  SetNextWakeupOp::create(builder, moduleOp.getLoc(), storageArg, noWakeup);

  // Create the `arc.initial` op to contain the ops for the initialization
  // phase.
  auto initialOp = InitialOp::create(builder, moduleOp.getLoc());
  initialBuilder.setInsertionPointToStart(&initialOp.getBody().emplaceBlock());

  // Create the `arc.final` op to contain the ops for the finalization phase.
  auto finalOp = FinalOp::create(builder, moduleOp.getLoc());
  finalBuilder.setInsertionPointToStart(&finalOp.getBody().emplaceBlock());

  // Anchor the `Phase::Old` section: old-phase values accumulate between the
  // `arc.final` op and this anchor, i.e. before every eval-phase op. The
  // anchor is a dead constant erased by the trailing cleanup.
  auto oldAnchor = hw::ConstantOp::create(builder, moduleOp.getLoc(),
                                          builder.getI1Type(), 0);
  oldBuilder.setInsertionPoint(oldAnchor);

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

  // Collect the signal-initializer value closure (see `sigInitClosure`):
  // seed with every `llhd.sig` init operand, then expand forward through
  // module-level ops deriving values from it (GEPs, casts).
  for (auto sigOp : moduleOp.getBodyBlock()->getOps<llhd::SignalOp>())
    if (auto init = sigOp.getInit())
      sigInitClosure.insert(init);
  if (!sigInitClosure.empty()) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &op : moduleOp.getOps()) {
        if (op.getNumResults() == 0)
          continue;
        if (llvm::any_of(op.getOperands(), [&](Value operand) {
              return sigInitClosure.contains(operand);
            }))
          for (auto result : op.getResults())
            changed |= sigInitClosure.insert(result).second;
      }
    }
  }

  // Classify which module ops must lower at the END of the eval body, after
  // every coroutine instance and signal drive has run: clocked state elements
  // (registers, memories, clocked DPI/terminate/trigger) detect their clock
  // edge against the post-process value of the CURRENT evaluation while their
  // data operands come from the pinned pre-eval `Phase::Old` section. This
  // pairing implements LRM scheduling-region semantics: a register clocks in
  // the evaluation in which its clock edge physically occurs, sampling values
  // from before any same-edge process writes (NBA stores commit after the
  // active region). Side-effecting consumers of register outputs (drives of
  // register-fed signals, taps, module outputs) defer with them so they
  // observe the committed values. Coroutine instances never defer: their
  // arguments are old-phase samples and their execution order among each
  // other must follow module order.
  DenseSet<Value> regClosure;
  for (auto &op : moduleOp.getOps()) {
    bool seed = isa<StateOp, MemoryOp, MemoryReadPortOp>(op);
    if (auto dpiOp = dyn_cast<sim::DPICallOp>(&op))
      seed = !!dpiOp.getClock();
    if (seed)
      for (auto result : op.getResults())
        regClosure.insert(result);
  }
  {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &op : moduleOp.getOps()) {
        if (op.getNumResults() == 0 || isa<CoroutineInstanceOp>(op))
          continue;
        if (llvm::any_of(op.getOperands(), [&](Value operand) {
              return regClosure.contains(operand);
            }))
          for (auto result : op.getResults())
            changed |= regClosure.insert(result).second;
      }
    }
  }
  auto deferredToEnd = [&](Operation &op) {
    if (isa<StateOp, MemoryOp, sim::ClockedTerminateOp, hw::TriggeredOp,
            hw::OutputOp>(op))
      return true;
    if (auto dpiOp = dyn_cast<sim::DPICallOp>(&op); dpiOp && dpiOp.getClock())
      return true;
    if (isa<CoroutineInstanceOp>(op))
      return false;
    return llvm::any_of(op.getOperands(), [&](Value operand) {
      return regClosure.contains(operand);
    });
  };
  auto skipInWalk = [](Operation &op) {
    if (mlir::isMemoryEffectFree(&op) &&
        !isa<hw::OutputOp, sim::ClockedTerminateOp>(op))
      return true;
    // Handled as part of `MemoryOp`.
    return isa<MemoryReadPortOp, MemoryWritePortOp>(op);
  };

  // Lower the ops: first the mid section (coroutine instances, drives, and
  // other effects not fed by clocked state), then the end section (clocked
  // state elements and their dependent effects). A mid-section op demanding a
  // register's new value still pulls that register's lowering forward through
  // the worklist; data stays consistent (old reads are pinned) and only that
  // register's edge-detection point degrades to the demand position.
  for (auto &op : moduleOp.getOps()) {
    if (skipInWalk(op) || deferredToEnd(op))
      continue;
    if (failed(lowerOp(&op)))
      return failure();
  }
  for (auto &op : moduleOp.getOps()) {
    if (skipInWalk(op) || !deferredToEnd(op))
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

  // Pick in which phases the given operation has to perform some work.
  SmallVector<Phase, 2> phases = {Phase::New};
  if (isa<seq::InitialOp>(op))
    phases = {Phase::Initial};
  if (isa<llhd::FinalOp>(op))
    phases = {Phase::Final};
  if (isa<StateOp>(op))
    phases = {Phase::Initial, Phase::New};
  if (isa<llhd::SignalOp>(op))
    phases = {Phase::Initial};

  // Side-effecting module-level ops consuming OR producing
  // signal-initializer values (the initializer computation itself plus the
  // object memset and typeinfo/property stores from class `new` at module
  // scope) are one-time initialization actions; see `sigInitClosure`.
  if (!isa<StateOp, sim::DPICallOp, MemoryOp, TapOp, InstanceOp,
           CoroutineInstanceOp, hw::TriggeredOp, hw::OutputOp, seq::InitialOp,
           llhd::FinalOp, llhd::CurrentTimeOp, llhd::SignalOp, llhd::DriveOp,
           sim::ClockedTerminateOp>(op) &&
      !mlir::isMemoryEffectFree(op) &&
      (llvm::any_of(
           op->getOperands(),
           [&](Value operand) { return sigInitClosure.contains(operand); }) ||
       llvm::any_of(op->getResults(), [&](Value result) {
         return sigInitClosure.contains(result);
       })))
    phases = {Phase::Initial};

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

  // Create the allocation op. Signal references allocate a slot of the
  // referenced type: the signal's storage.
  auto allocType = result.getType();
  if (auto refType = dyn_cast<llhd::RefType>(allocType))
    allocType = refType.getNestedType();
  auto alloc = AllocStateOp::create(allocBuilder, result.getLoc(),
                                    StateType::get(allocType), storageArg);
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
  StateWriteOp::create(builder, loc, oldStorage, clock);

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
    return oldBuilder;
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
      .Case<StateOp, sim::DPICallOp, MemoryOp, TapOp, InstanceOp,
            CoroutineInstanceOp, hw::TriggeredOp, hw::OutputOp, seq::InitialOp,
            llhd::FinalOp, llhd::CurrentTimeOp, llhd::SignalOp, llhd::DriveOp,
            sim::ClockedTerminateOp>([&](auto op) { return lower(op); })

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
      StateWriteOp::create(module.initialBuilder, value.getLoc(), state, value);
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
      StateWriteOp::create(module.builder, loweredReset.getLoc(), state, value);
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
    StateWriteOp::create(module.builder, value.getLoc(), state, value);

  // Since we just wrote the new state value to storage, insert read ops in the
  // pinned old section that keep the old value around for any later ops that
  // still need it. The old section executes before every state update, so the
  // reads observe the pre-eval value and dominate all later users.
  for (auto [state, result] : llvm::zip(states, results)) {
    auto oldValue =
        StateReadOp::create(module.oldBuilder, result.getLoc(), state);
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
    MemoryWriteOp::create(module.builder, write.getLoc(), state, address, data);
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
  StateWriteOp::create(module.builder, op.getLoc(), state, value);
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
    StateWriteOp::create(module.builder, value.getLoc(), state, value);
  }

  // HACK: Also ensure that storage has been allocated for all outputs.
  // Otherwise only the actually used instance outputs would be allocated, which
  // would make the optimization user-visible. Remove this once we use the debug
  // dialect.
  for (auto result : op.getResults())
    module.getAllocatedState(result);

  return success();
}

/// Lower a coroutine instance.
///
/// An `arc.coroutine.instance` runs a top-level coroutine continuously inside a
/// model. The coroutine's program counter, local state, and next wakeup time
/// are kept in persistent state slots, and the values it yields are latched
/// into result slots so they remain readable on evaluations where the coroutine
/// does not run. On every evaluation the instance re-enters the coroutine if
/// its scheduled wakeup time has been reached, stores the resulting program
/// counter, state, and yielded values, and folds the next wakeup time into the
/// model's global wakeup schedule.
///
/// A coroutine that has halted or returned must never be re-entered. Instead of
/// inspecting the program counter on entry, the lowering forces the stored
/// wakeup time to `UINT64_MAX` ("never") as soon as the coroutine reports a
/// halt or return, so the time guard alone keeps it suspended.
LogicalResult OpLowering::lower(CoroutineInstanceOp op) {
  assert(phase == Phase::New);

  // A coroutine samples its arguments in the New phase, so that a re-entry sees
  // the up-to-date values produced in the same evaluation and the change
  // detector below compares against fresh values.
  SmallVector<Value> inputs;
  for (auto input : op.getArgs())
    inputs.push_back(lowerValue(input, Phase::New));
  if (initial)
    return success();
  if (llvm::is_contained(inputs, Value{}))
    return failure();

  // Resolve the callee to obtain its state, program counter, and result types.
  // The callee's last result is the next wakeup time; it is consumed for
  // scheduling and not exposed as a result of the instance.
  auto callee = op.getCalleeAttr();
  auto defineOp =
      module.symbolTable.lookup<CoroutineDefineOp>(callee.getAttr());
  assert(defineOp && "verified by CoroutineInstanceOp::verifySymbolUses");
  auto loc = op.getLoc();
  auto *context = op.getContext();
  auto stateType = CoroutineStateType::get(context, callee);
  auto pcType = CoroutinePCType::get(context, callee);
  auto i64Type = module.builder.getI64Type();

  // Allocate the persistent program counter, state, and wakeup slots. Their
  // zero-initialized contents represent the coroutine's start program counter,
  // an unread initial state, and a wakeup time of zero ("run immediately").
  auto pcSlot = AllocStateOp::create(module.allocBuilder, loc,
                                     StateType::get(pcType), module.storageArg);
  auto stateSlot = AllocStateOp::create(
      module.allocBuilder, loc, StateType::get(stateType), module.storageArg);
  auto wakeupSlot = AllocStateOp::create(
      module.allocBuilder, loc, StateType::get(i64Type), module.storageArg);

  // Allocate a slot for each yielded value so that it persists across
  // evaluations where the coroutine does not run.
  SmallVector<Value> resultSlots;
  for (auto result : op.getResults()) {
    auto slot = module.getAllocatedState(result);
    if (!slot)
      return failure();
    resultSlots.push_back(slot);
  }

  // Detect changes on the observed arguments: each argument's value from the
  // previous evaluation is held in a state slot, and a change is an inequality
  // against the freshly sampled value. The observe bitmask reported by the
  // coroutine on its last run selects which arguments matter; unobserved
  // changes are ignored. The previous-value slots are updated unconditionally
  // so they always track the latest value.
  Value maskSlot;
  Value anyChange = hw::ConstantOp::create(module.builder, loc,
                                           module.builder.getI1Type(), 0);
  if (!inputs.empty()) {
    auto maskType = module.builder.getIntegerType(inputs.size());
    maskSlot = AllocStateOp::create(
        module.allocBuilder, loc, StateType::get(maskType), module.storageArg);
    auto mask = StateReadOp::create(module.builder, loc, maskSlot);
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (!op.getSensitivityMask()[index])
        continue;
      auto prevSlot = AllocStateOp::create(module.allocBuilder, loc,
                                           StateType::get(input.getType()),
                                           module.storageArg);
      auto prev = StateReadOp::create(module.builder, loc, prevSlot);
      StateWriteOp::create(module.builder, loc, prevSlot, input);
      auto changed = comb::ICmpOp::create(module.builder, loc,
                                          comb::ICmpPredicate::ne, input, prev);
      auto maskBit =
          comb::ExtractOp::create(module.builder, loc, mask,
                                  static_cast<unsigned>(index), /*bitWidth=*/1);
      auto masked = comb::AndOp::create(module.builder, loc, changed, maskBit);
      anyChange = comb::OrOp::create(module.builder, loc, anyChange, masked);
    }
  }

  // Re-enter the coroutine if its scheduled wakeup time has been reached or if
  // an observed argument changed.
  auto now = CurrentTimeOp::create(module.builder, loc, module.storageArg);
  auto wakeup = StateReadOp::create(module.builder, loc, wakeupSlot);
  auto timeReady = comb::ICmpOp::create(module.builder, loc,
                                        comb::ICmpPredicate::uge, now, wakeup);
  auto ready = comb::OrOp::create(module.builder, loc, timeReady, anyChange);
  auto ifOp =
      scf::IfOp::create(module.builder, loc, ready, /*withElseRegion=*/false);
  {
    OpBuilder::InsertionGuard guard(module.builder);
    module.builder.setInsertionPoint(ifOp.thenYield());

    auto oldState = StateReadOp::create(module.builder, loc, stateSlot);
    auto oldPc = StateReadOp::create(module.builder, loc, pcSlot);

    // The call returns the resume state and program counter followed by the
    // coroutine's own results, the last of which is the next wakeup time.
    SmallVector<Type> callResultTypes;
    callResultTypes.push_back(stateType);
    callResultTypes.push_back(pcType);
    llvm::append_range(callResultTypes, defineOp.getResultTypes());
    auto call = CoroutineCallOp::create(module.builder, loc, callResultTypes,
                                        callee, oldState, oldPc, inputs);
    auto newState = call.getResult(0);
    auto newPc = call.getResult(1);
    auto wakeupNew = call.getResults().back();
    auto maskNew = call.getResult(2 + op.getNumResults());

    // Force the wakeup time to "never" once the coroutine halts or returns, so
    // the time guard above prevents it from ever being re-entered.
    auto isHalt = CoroutinePCIsHaltOp::create(module.builder, loc, newPc);
    auto isReturn = CoroutinePCIsReturnOp::create(module.builder, loc, newPc);
    auto isDone = comb::OrOp::create(module.builder, loc, isHalt, isReturn);
    auto never = hw::ConstantOp::create(module.builder, loc, i64Type, -1);
    auto wakeupEff =
        comb::MuxOp::create(module.builder, loc, isDone, never, wakeupNew);

    StateWriteOp::create(module.builder, loc, stateSlot, newState);
    StateWriteOp::create(module.builder, loc, pcSlot, newPc);
    StateWriteOp::create(module.builder, loc, wakeupSlot, wakeupEff);
    if (maskSlot)
      StateWriteOp::create(module.builder, loc, maskSlot, maskNew);
    for (auto [index, slot] : llvm::enumerate(resultSlots))
      StateWriteOp::create(module.builder, loc, slot,
                           call.getResult(2 + index));
  }

  // Fold the coroutine's pending wakeup time into the model's wakeup schedule.
  // This runs unconditionally: even when the coroutine did not execute this
  // evaluation, its stored wakeup must keep the model scheduled.
  auto curWakeup = StateReadOp::create(module.builder, loc, wakeupSlot);
  auto nextWakeup =
      GetNextWakeupOp::create(module.builder, loc, module.storageArg);
  auto minWakeup =
      arith::MinUIOp::create(module.builder, loc, curWakeup, nextWakeup);
  SetNextWakeupOp::create(module.builder, loc, module.storageArg, minWakeup);

  return success();
}

/// Lower `hw.triggered` by inlining its body under a posedge check.
LogicalResult OpLowering::lower(hw::TriggeredOp op) {
  assert(phase == Phase::New);

  if (op.getEvent() != hw::EventControl::AtPosEdge) {
    if (!initial)
      return op.emitOpError("only posedge triggers are supported");
    return success();
  }

  lowerValue(op.getTrigger(), Phase::New);
  SmallVector<Value> inputs;
  for (auto input : op.getInputs())
    inputs.push_back(lowerValue(input, Phase::Old));
  if (initial)
    return success();
  if (llvm::is_contained(inputs, Value{}))
    return failure();

  auto ifClockOp = createIfClockOp(op.getTrigger());
  if (!ifClockOp)
    return failure();

  OpBuilder::InsertionGuard guard(module.builder);
  module.builder.setInsertionPoint(ifClockOp.thenYield());

  // Expose the trigger inputs as values for the body block arguments.
  for (auto [arg, input] : llvm::zip(op.getBodyBlock()->getArguments(), inputs))
    module.loweredValues[{arg, Phase::New}] = input;
  for (auto &bodyOp : llvm::make_early_inc_range(*op.getBodyBlock())) {
    OpLowering bodyLowering(&bodyOp, Phase::New, module);
    bodyLowering.initial = false;
    if (failed(bodyLowering.lower()))
      return failure();
  }

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
    StateWriteOp::create(module.builder, value.getLoc(), state, value);
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

  // Lower each op in the body. We maintain a mapping from original values
  // defined in the body to their cloned counterparts.
  IRMapping bodyMapping;
  auto *initialBlock = module.initialBuilder.getBlock();

  // Pre-lower all llhd.current_time ops inside the body. This reuses the
  // existing lower(llhd::CurrentTimeOp) logic which handles Phase::Initial
  // by replacing with constant 0 time.
  auto result = op.walk([&](llhd::CurrentTimeOp timeOp) {
    if (failed(lower(timeOp)))
      return WalkResult::interrupt();
    auto loweredTime = module.loweredValues.lookup({timeOp.getResult(), phase});
    timeOp.replaceAllUsesWith(loweredTime);
    timeOp.erase();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  for (auto &bodyOp : op.getOps()) {
    if (isa<seq::YieldOp>(bodyOp))
      continue;

    // Clone the operation.
    auto *clonedOp = module.initialBuilder.clone(bodyOp, bodyMapping);
    auto result = clonedOp->walk([&](Operation *nestedClonedOp) {
      for (auto &operand : nestedClonedOp->getOpOperands()) {
        // Skip operands defined within the cloned tree.
        if (clonedOp->isAncestor(operand.get().getParentBlock()->getParentOp()))
          continue;
        // Skip operands defined within the initial block (e.g., results of
        // previously lowered ops like our zeroTime).
        if (auto *defOp = operand.get().getDefiningOp())
          if (defOp->getBlock() == initialBlock)
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

    // Keep track of the results in both mappings.
    for (auto [result, lowered] :
         llvm::zip(bodyOp.getResults(), clonedOp->getResults())) {
      bodyMapping.map(result, lowered);
      module.loweredValues[{result, Phase::Initial}] = lowered;
    }
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

  // Pre-lower all llhd.current_time ops inside the body. This reuses the
  // existing lower(llhd::CurrentTimeOp) logic which handles Phase::Final
  // by replacing with arc.current_time.
  auto result = op.walk([&](llhd::CurrentTimeOp timeOp) {
    if (failed(lower(timeOp)))
      return WalkResult::interrupt();
    auto loweredTime = module.loweredValues.lookup({timeOp.getResult(), phase});
    timeOp.replaceAllUsesWith(loweredTime);
    timeOp.erase();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

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
  executeOp.walk([&](llhd::HaltOp haltOp) {
    auto builder = OpBuilder(haltOp);
    scf::YieldOp::create(builder, haltOp.getLoc());
    haltOp.erase();
  });

  return success();
}

/// Lower `llhd.current_time` based on the current phase:
/// - Phase::Initial: Replace with constant 0 time.
/// - Phase::Old, Phase::New, Phase::Final: Replace with `arc.current_time`
///   followed by `llhd.int_to_time`.
LogicalResult OpLowering::lower(llhd::CurrentTimeOp op) {
  if (initial)
    return success();

  auto loc = op.getLoc();
  Value time;

  switch (phase) {
  case Phase::Initial: {
    // During initialization, time is always 0.
    auto zeroInt = hw::ConstantOp::create(
        module.initialBuilder, loc, module.initialBuilder.getI64Type(), 0);
    time = llhd::IntToTimeOp::create(module.initialBuilder, loc, zeroInt);
    break;
  }
  case Phase::Old:
  case Phase::New:
  case Phase::Final: {
    // Get the current time from storage.
    auto &builder = module.getBuilder(phase);
    auto timeInt = CurrentTimeOp::create(builder, loc, module.storageArg);
    time = llhd::IntToTimeOp::create(builder, loc, timeInt);
    break;
  }
  }

  module.loweredValues[{op.getResult(), phase}] = time;
  return success();
}

/// Lower a module-level signal definition: allocate persistent storage of the
/// referenced type and write the init value in the initial phase. Probes and
/// drives of the signal read/write the storage; see their lowerings below.
LogicalResult OpLowering::lower(llhd::SignalOp op) {
  assert(phase == Phase::Initial);
  if (initial) {
    lowerValue(op.getInit(), Phase::Initial);
    return success();
  }
  auto value = lowerValue(op.getInit(), Phase::Initial);
  if (!value)
    return failure();
  auto state = module.getAllocatedState(op->getResult(0));
  if (!state)
    return failure();
  StateWriteOp::create(module.initialBuilder, op.getLoc(), state, value);
  return success();
}

/// Lower a module-level drive: a write of the driven value to the signal's
/// storage. Only zero-real-time (delta/epsilon) delays are supported; they
/// express update ordering, which the state-slot model resolves per
/// evaluation. Drives with real-time delays would need a scheduler queue and
/// are rejected.
LogicalResult OpLowering::lower(llhd::DriveOp op) {
  assert(phase == Phase::New);

  auto sigResult = dyn_cast<OpResult>(op.getSignal());
  // A drive of a constant-offset bit-slice of a module-level signal
  // (`llhd.drv (llhd.sig.extract %sig from K), %v`) lowers as a
  // read-modify-write splice into the parent signal's storage.
  llhd::SigExtractOp sliceOp;
  uint64_t sliceOffset = 0;
  if (sigResult) {
    if (auto extract = dyn_cast<llhd::SigExtractOp>(sigResult.getOwner())) {
      if (auto cst = extract.getLowBit().getDefiningOp<hw::ConstantOp>()) {
        auto parent = dyn_cast<OpResult>(extract.getInput());
        if (parent && isa<llhd::SignalOp>(parent.getOwner())) {
          sliceOp = extract;
          sliceOffset = cst.getValue().getZExtValue();
          sigResult = parent;
        }
      }
    }
  }
  // A drive of an array element of a module-level signal
  // (`llhd.drv (llhd.sig.array_get %sig[%idx]), %v`) lowers as a
  // read-modify-write of the parent signal's storage through
  // `hw.array_inject` -- the array-typed sibling of the bit-slice splice.
  llhd::SigArrayGetOp elementOp;
  if (sigResult) {
    if (auto get = dyn_cast<llhd::SigArrayGetOp>(sigResult.getOwner())) {
      auto parent = dyn_cast<OpResult>(get.getInput());
      if (parent && isa<llhd::SignalOp>(parent.getOwner())) {
        elementOp = get;
        sigResult = parent;
      }
    }
  }
  if (!sigResult || !isa<llhd::SignalOp>(sigResult.getOwner())) {
    if (!initial)
      return op.emitOpError()
             << "drive of a reference that is not a module-level signal is "
                "not supported";
    return success();
  }
  auto timeOp = op.getTime().getDefiningOp<llhd::ConstantTimeOp>();
  if (!timeOp || timeOp.getValue().getTime() != 0) {
    if (!initial)
      return op.emitOpError()
             << "drive with a nonzero real-time delay is not supported";
    return success();
  }

  if (initial) {
    lowerValue(op.getValue(), Phase::New);
    if (op.getEnable())
      lowerValue(op.getEnable(), Phase::New);
    if (elementOp)
      lowerValue(elementOp.getIndex(), Phase::New);
    return success();
  }

  auto value = lowerValue(op.getValue(), Phase::New);
  if (!value)
    return failure();
  auto state = module.getAllocatedState(sigResult);
  if (!state)
    return failure();

  // Slice drive: splice the driven bits into the current storage value.
  if (sliceOp) {
    auto valueIntTy = dyn_cast<IntegerType>(value.getType());
    auto curTy = dyn_cast<IntegerType>(
        cast<StateType>(state.getType()).getType());
    if (!valueIntTy || !curTy)
      return op.emitOpError()
             << "slice drive of a non-integer module-level signal is not "
                "supported";
    unsigned parentWidth = curTy.getWidth();
    unsigned sliceWidth = valueIntTy.getWidth();
    if (sliceOffset + sliceWidth > parentWidth)
      return op.emitOpError() << "slice drive out of range";
    auto &builder = module.builder;
    Value cur = StateReadOp::create(builder, op.getLoc(), state);
    SmallVector<Value, 3> pieces; // MSB-first for comb.concat
    if (sliceOffset + sliceWidth < parentWidth)
      pieces.push_back(comb::ExtractOp::create(
          builder, op.getLoc(), cur, sliceOffset + sliceWidth,
          parentWidth - sliceOffset - sliceWidth));
    pieces.push_back(value);
    if (sliceOffset > 0)
      pieces.push_back(
          comb::ExtractOp::create(builder, op.getLoc(), cur, 0, sliceOffset));
    value = pieces.size() == 1
                ? pieces.front()
                : Value(comb::ConcatOp::create(builder, op.getLoc(), pieces));
  }

  // Element drive: read-modify-write of the parent array signal's storage.
  if (elementOp) {
    auto index = lowerValue(elementOp.getIndex(), Phase::New);
    if (!index)
      return failure();
    Value cur = StateReadOp::create(module.builder, op.getLoc(), state);
    value = hw::ArrayInjectOp::create(module.builder, op.getLoc(), cur, index,
                                      value);
  }

  if (op.getEnable()) {
    auto enable = lowerValue(op.getEnable(), Phase::New);
    if (!enable)
      return failure();
    auto ifOp = scf::IfOp::create(module.builder, op.getLoc(), enable,
                                  /*withElseRegion=*/false);
    OpBuilder::InsertionGuard guard(module.builder);
    module.builder.setInsertionPoint(ifOp.thenYield());
    StateWriteOp::create(module.builder, op.getLoc(), state, value);
    return success();
  }
  StateWriteOp::create(module.builder, op.getLoc(), state, value);
  return success();
}

/// Handle probes of module-level signals: a read of the signal's storage.
Value OpLowering::lowerValue(llhd::ProbeOp op, OpResult result, Phase phase) {
  auto sigResult = dyn_cast<OpResult>(op.getSignal());
  // A probe of a constant-offset bit-slice of a module-level signal
  // (`llhd.prb (llhd.sig.extract %sig from K)`) lowers as a read of the
  // parent signal's storage plus an extract of the slice -- the probe-side
  // mirror of the slice-drive splice above.
  llhd::SigExtractOp sliceOp;
  uint64_t sliceOffset = 0;
  if (sigResult) {
    if (auto extract = dyn_cast<llhd::SigExtractOp>(sigResult.getOwner())) {
      if (auto cst = extract.getLowBit().getDefiningOp<hw::ConstantOp>()) {
        auto parent = dyn_cast<OpResult>(extract.getInput());
        if (parent && isa<llhd::SignalOp>(parent.getOwner())) {
          sliceOp = extract;
          sliceOffset = cst.getValue().getZExtValue();
          sigResult = parent;
        }
      }
    }
  }
  if (!sigResult || !isa<llhd::SignalOp>(sigResult.getOwner())) {
    if (!initial)
      emitError(op.getLoc())
          << "probe of a reference that is not a module-level signal is not "
             "supported";
    return {};
  }
  if (initial)
    return {};
  auto state = module.getAllocatedState(sigResult);
  if (!state)
    return {};
  Value read =
      StateReadOp::create(module.getBuilder(phase), op.getLoc(), state);
  if (sliceOp) {
    auto resTy = dyn_cast<IntegerType>(op.getResult().getType());
    auto curTy =
        dyn_cast<IntegerType>(cast<StateType>(state.getType()).getType());
    if (!resTy || !curTy || sliceOffset + resTy.getWidth() > curTy.getWidth()) {
      emitError(op.getLoc())
          << "slice probe of a non-integer module-level signal is not "
             "supported";
      return {};
    }
    read = comb::ExtractOp::create(module.getBuilder(phase), op.getLoc(), read,
                                   sliceOffset, resTy.getWidth());
  }
  return read;
}

LogicalResult OpLowering::lower(sim::ClockedTerminateOp op) {
  if (phase != Phase::New)
    return success();

  if (initial)
    return success();

  auto ifClockOp = createIfClockOp(op.getClock());
  if (!ifClockOp)
    return failure();

  OpBuilder::InsertionGuard guard(module.builder);
  module.builder.setInsertionPoint(ifClockOp.thenYield());

  auto loc = op.getLoc();
  Value cond = lowerValue(op.getCondition(), phase);
  if (!cond)
    return op.emitOpError("Failed to lower condition");

  auto ifOp = createOrReuseIf(module.builder, cond, false);
  if (!ifOp)
    return op.emitOpError("Failed to create condition block");

  module.builder.setInsertionPoint(ifOp.thenYield());

  arc::TerminateOp::create(module.builder, loc, module.storageArg,
                           op.getSuccessAttr());

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
  // Check if the value has already been lowered.
  if (auto lowered = module.loweredValues.lookup({value, phase}))
    return lowered;

  // Handle module inputs. They read the same in all phases.
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() != module.moduleOp.getBodyBlock()) {
      if (!initial)
        emitError(arg.getLoc()) << "block argument has not been lowered";
      return {};
    }
    if (initial)
      return {};
    auto state = module.allocatedInputs[arg.getArgNumber()];
    return StateReadOp::create(module.getBuilder(phase), arg.getLoc(), state);
  }

  // At this point the value is the result of an op. (Block arguments are
  // handled above.)
  auto result = cast<OpResult>(value);
  auto *op = result.getOwner();

  // Special handling for some ops.
  if (auto instOp = dyn_cast<InstanceOp>(op))
    return lowerValue(instOp, result, phase);
  if (auto instOp = dyn_cast<CoroutineInstanceOp>(op))
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
  if (auto probeOp = dyn_cast<llhd::ProbeOp>(op))
    return lowerValue(probeOp, result, phase);
  // Regions cloned into the model (e.g. `llhd.final` bodies) may reference a
  // module-level signal by its reference value. Hand out the signal's
  // storage: both are pointers into the model storage representationally;
  // the bridging cast cancels out during the LLVM lowering.
  if (auto sigOp = dyn_cast<llhd::SignalOp>(op)) {
    if (initial)
      return {};
    auto state = module.getAllocatedState(result);
    if (!state)
      return {};
    return mlir::UnrealizedConversionCastOp::create(
               module.getBuilder(phase), sigOp.getLoc(), result.getType(),
               ValueRange{state})
        .getResult(0);
  }

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

/// Handle the yielded values of a coroutine instance. The values are latched
/// into a result slot by the instance lowering; reading the new value requires
/// the instance to be lowered first so the slot has been written, while reading
/// the old value observes the slot's contents before this evaluation's update.
Value OpLowering::lowerValue(CoroutineInstanceOp op, OpResult result,
                             Phase phase) {
  if (initial) {
    // The instance only ever runs in the new phase, where it writes the result
    // slots. Make sure that has happened before we read them.
    if (phase == Phase::New)
      addPending(op, Phase::New);
    return {};
  }

  // Old-value reads land in the pinned old section, which executes before the
  // instance updates its result slots, so they are sound even when the
  // instance's new value has already been lowered.

  auto state = module.getAllocatedState(result);
  return StateReadOp::create(module.getBuilder(phase), result.getLoc(), state);
}

/// Handle uses of a state. This creates an `arc.state_read` op to read from the
/// state's storage. If the new value after all updates is requested, marks the
/// state as to be lowered first (which will perform the writes). Old-value
/// reads are emitted into the pinned old section ahead of all writes.
Value OpLowering::lowerValue(StateOp op, OpResult result, Phase phase) {
  if (initial) {
    // Ensure that the new or initial value has been written by the lowering of
    // the state op before we attempt to read it.
    if (phase == Phase::New || phase == Phase::Initial)
      addPending(op, phase);
    return {};
  }

  // Old-value reads land in the pinned old section, which executes before the
  // state's write, so they are sound even when the new value has already been
  // lowered.

  auto state = module.getAllocatedState(result);
  return StateReadOp::create(module.getBuilder(phase), result.getLoc(), state);
}

/// Handle uses of a DPI call. This creates an `arc.state_read` op to read from
/// the state's storage. If the new value after all updates is requested, marks
/// the state as to be lowered first (which will perform the writes). Old-value
/// reads are emitted into the pinned old section ahead of all writes.
Value OpLowering::lowerValue(sim::DPICallOp op, OpResult result, Phase phase) {
  if (initial) {
    // Ensure that the new or initial value has been written by the lowering of
    // the state op before we attempt to read it.
    if (phase == Phase::New || phase == Phase::Initial)
      addPending(op, phase);
    return {};
  }

  // Old-value reads land in the pinned old section, which executes before the
  // state's write, so they are sound even when the new value has already been
  // lowered.

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

  // Old-value reads land in the pinned old section, which executes before all
  // memory writes, so they are sound even when the writes have already been
  // lowered.
  assert(phase == Phase::Old || phase == Phase::New);

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
    StateWriteOp::create(module.initialBuilder, value.getLoc(), state, value);
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
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto moduleOp : llvm::make_early_inc_range(op.getOps<HWModuleOp>())) {
    if (failed(ModuleLowering(moduleOp, symbolTable).run()))
      return signalPassFailure();
    moduleOp.erase();
  }

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
