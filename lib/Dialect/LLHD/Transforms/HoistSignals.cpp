//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-hoist-signals"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_HOISTSIGNALSPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::PointerUnion;
using llvm::SmallSetVector;

//===----------------------------------------------------------------------===//
// Probe Hoisting
//===----------------------------------------------------------------------===//

namespace {
/// The struct performing the hoisting of probes in a single region.
struct ProbeHoister {
  ProbeHoister(Region &region) : region(region) {}
  void hoist();

  void findValuesLiveAcrossWait(Liveness &liveness);
  void hoistProbes();

  /// The region we are hoisting ops out of.
  Region &region;

  /// The set of values that are alive across wait ops themselves, or that have
  /// transitive users that are live across wait ops.
  DenseSet<Value> liveAcrossWait;

  /// A lookup table of probes we have already hoisted, for deduplication.
  DenseMap<Value, ProbeOp> hoistedProbes;
};
} // namespace

void ProbeHoister::hoist() {
  Liveness liveness(region.getParentOp());
  findValuesLiveAcrossWait(liveness);
  hoistProbes();
}

/// Find all values in the region that are alive across `llhd.wait` operations,
/// or that have transitive uses that are alive across waits. We can only hoist
/// probes that do not feed data flow graphs that are alive across such wait
/// ops. Since control flow edges in `cf.br` and `cf.cond_br` ops are
/// side-effect free, we have no guarantee that moving a probe out of a process
/// could potentially cause other ops to become eligible for a move out of the
/// process. Therefore, if such ops are moved outside of the process, they are
/// effectively moved across the waits and thus sample their operands at
/// different points in time. Only values that are explicitly carried across
/// `llhd.wait`, where the LLHD dialect has control over the control flow
/// semantics, may have probes in their fan-in cone hoisted out.
void ProbeHoister::findValuesLiveAcrossWait(Liveness &liveness) {
  // First find all values that are live across `llhd.wait` operations. We are
  // only interested in values defined in the current region.
  SmallVector<Value> worklist;
  for (auto &block : region)
    if (isa<WaitOp>(block.getTerminator()))
      for (auto value : liveness.getLiveOut(&block))
        if (value.getParentRegion() == &region)
          if (liveAcrossWait.insert(value).second)
            worklist.push_back(value);

  // Propagate liveness information along the use-def chain and across control
  // flow. This will allow us to check `liveAcrossWait` to know if a value
  // escapes across a wait along its use-def chain that isn't an explicit
  // successor operand of the wait op.
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (auto *defOp = value.getDefiningOp()) {
      for (auto operand : defOp->getOperands())
        if (operand.getParentRegion() == &region)
          if (liveAcrossWait.insert(operand).second)
            worklist.push_back(operand);
    } else {
      auto blockArg = cast<BlockArgument>(value);
      for (auto &use : blockArg.getOwner()->getUses()) {
        auto branch = dyn_cast<BranchOpInterface>(use.getOwner());
        if (!branch)
          continue;
        auto operand = branch.getSuccessorOperands(
            use.getOperandNumber())[blockArg.getArgNumber()];
        if (operand.getParentRegion() == &region)
          if (liveAcrossWait.insert(operand).second)
            worklist.push_back(operand);
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << liveAcrossWait.size()
                          << " values live across wait\n");
}

/// Hoist any probes at the beginning of resuming blocks out of the process if
/// their values do not leak across wait ops. Resuming blocks are blocks where
/// all predecessors are `llhd.wait` ops, and the entry block. Only waits
/// without any side-effecting op in between themselves and the beginning of the
/// block can be hoisted.
void ProbeHoister::hoistProbes() {
  auto findExistingProbe = [&](Value signal) {
    for (auto *user : signal.getUsers())
      if (auto probeOp = dyn_cast<ProbeOp>(user))
        if (probeOp->getParentRegion()->isProperAncestor(&region))
          return probeOp;
    return ProbeOp{};
  };

  for (auto &block : region) {
    // We can only hoist probes in blocks where all predecessors have wait
    // terminators.
    if (!llvm::all_of(block.getPredecessors(), [](auto *predecessor) {
          return isa<WaitOp>(predecessor->getTerminator());
        }))
      continue;

    for (auto &op : llvm::make_early_inc_range(block)) {
      auto probeOp = dyn_cast<ProbeOp>(op);

      // We can only hoist probes that have no side-effecting ops between
      // themselves and the beginning of a block. If we see a side-effecting op,
      // give up on this block.
      if (!probeOp) {
        if (isMemoryEffectFree(&op))
          continue;
        else
          break;
      }

      // Only hoist probes that don't leak across wait ops.
      if (liveAcrossWait.contains(probeOp)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping (live across wait) " << probeOp << "\n");
        continue;
      }

      // We can only hoist probes of signals that are declared outside the
      // process.
      if (!probeOp.getSignal().getParentRegion()->isProperAncestor(&region)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping (local signal) " << probeOp << "\n");
        continue;
      }

      // Move the probe out of the process, trying to reuse any previous probe
      // that we've already hoisted.
      auto &hoistedOp = hoistedProbes[probeOp.getSignal()];
      if (hoistedOp) {
        LLVM_DEBUG(llvm::dbgs() << "- Replacing " << probeOp << "\n");
        probeOp.replaceAllUsesWith(hoistedOp.getResult());
        probeOp.erase();
      } else {
        LLVM_DEBUG(llvm::dbgs() << "- Hoisting " << probeOp << "\n");
        if (auto existingOp = findExistingProbe(probeOp.getSignal())) {
          probeOp.replaceAllUsesWith(existingOp.getResult());
          probeOp.erase();
          hoistedOp = existingOp;
        } else {
          if (auto *defOp = probeOp.getSignal().getDefiningOp())
            probeOp->moveAfter(defOp);
          else
            probeOp->moveBefore(region.getParentOp());
          hoistedOp = probeOp;
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Drive Operand Tracking
//===----------------------------------------------------------------------===//

namespace {
/// An operand value on a drive operation. Can represent constant `IntegerAttr`
/// or `TimeAttr` operands, regular `Value` operands, a typed dont-care value,
/// or a null value.
struct DriveValue {
  typedef PointerUnion<Value, IntegerAttr, TimeAttr, Type> Data;
  Data data;

  // Create a null `DriveValue`.
  DriveValue() : data(Value{}) {}

  // Create a don't care `DriveValue`.
  static DriveValue dontCare(Type type) { return DriveValue(type); }

  /// Create a `DriveValue` from a non-null constant `IntegerAttr`.
  DriveValue(IntegerAttr attr) : data(attr) {}

  /// Create a `DriveValue` from a non-null constant `TimeAttr`.
  DriveValue(TimeAttr attr) : data(attr) {}

  // Create a `DriveValue` from a non-null `Value`. If the value is defined by a
  // constant-like op, stores the constant attribute instead.
  DriveValue(Value value) : data(value) {
    Attribute attr;
    if (auto *defOp = value.getDefiningOp())
      if (m_Constant(&attr).match(defOp))
        TypeSwitch<Attribute>(attr).Case<IntegerAttr, TimeAttr>(
            [&](auto attr) { data = attr; });
  }

  bool operator==(const DriveValue &other) const { return data == other.data; }
  bool operator!=(const DriveValue &other) const { return data != other.data; }

  bool isDontCare() const { return isa<Type>(data); }
  explicit operator bool() const { return bool(data); }

  Type getType() const {
    return TypeSwitch<Data, Type>(data)
        .Case<Value, IntegerAttr, TimeAttr>([](auto x) { return x.getType(); })
        .Case<Type>([](auto type) { return type; });
  }

private:
  explicit DriveValue(Data data) : data(data) {}
};

/// The operands of an `llhd.drv`, represented as `DriveValue`s.
struct DriveOperands {
  DriveValue value;
  DriveValue delay;
  DriveValue enable;
};

/// A set of drives to a single slot. Tracks the drive operations, the operands
/// of the drive before each terminator, and which operands have a uniform
/// value.
struct DriveSet {
  /// The drive operations covered by the information in this struct.
  SmallPtrSet<Operation *, 2> ops;
  /// The drive operands at each terminator.
  SmallDenseMap<Operation *, DriveOperands, 2> operands;
  /// The drive operands that are uniform across all terminators, or null if
  /// non-uniform.
  DriveOperands uniform;
};
} // namespace

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const DriveValue &dv) {
  if (!dv.data)
    os << "null";
  else
    TypeSwitch<DriveValue::Data>(dv.data)
        .Case<Value, IntegerAttr, TimeAttr>([&](auto x) { os << x; })
        .Case<Type>([&](auto) { os << "dont-care"; });
  return os;
}

//===----------------------------------------------------------------------===//
// Drive Hoisting
//===----------------------------------------------------------------------===//

namespace {
/// The struct performing the hoisting of drives in a process.
struct DriveHoister {
  DriveHoister(ProcessOp processOp) : processOp(processOp) {}
  void hoist();

  void findHoistableSlots();
  void collectDriveSets();
  void finalizeDriveSets();
  void hoistDrives();

  /// The process we are hoisting drives out of.
  ProcessOp processOp;

  /// The slots for which we are trying to hoist drives. Mostly `llhd.sig` ops
  /// in practice. This establishes a deterministic order for slots, such that
  /// everything else in the pass can operate using unordered maps and sets.
  SmallSetVector<Value, 8> slots;
  SmallVector<Operation *> suspendOps;
  SmallDenseMap<Value, DriveSet> driveSets;
};
} // namespace

void DriveHoister::hoist() {
  findHoistableSlots();
  collectDriveSets();
  finalizeDriveSets();
  hoistDrives();
}

/// Identify any slots driven under the current region which are candidates for
/// hoisting. This checks if the slots escape or alias in any way which we
/// cannot reason about.
void DriveHoister::findHoistableSlots() {
  SmallPtrSet<Value, 8> seenSlots;
  processOp.walk([&](DriveOp op) {
    auto slot = op.getSignal();
    if (!seenSlots.insert(slot).second)
      return;

    // We can only hoist drives to slots declared by a `llhd.sig` op outside the
    // current region.
    if (!slot.getDefiningOp<llhd::SignalOp>())
      return;
    if (!slot.getParentRegion()->isProperAncestor(&processOp.getBody()))
      return;

    // Ensure the slot is not used in any way we cannot reason about.
    if (!llvm::all_of(slot.getUsers(), [&](auto *user) {
          // Ignore uses outside of the region.
          if (!processOp.getBody().isAncestor(user->getParentRegion()))
            return true;
          return isa<ProbeOp, DriveOp>(user);
        }))
      return;

    slots.insert(slot);
  });
  LLVM_DEBUG(llvm::dbgs() << "Found " << slots.size()
                          << " slots for drive hoisting\n");
}

/// Collect the operands of all hoistable drives into a per-slot drive set.
/// After this function returns, the `driveSets` contains a drive set for each
/// slot that has at least one hoistable drive. Each drive set lists the drive
/// operands for each suspending terminator. If a slot is not driven before a
/// terminator, the drive set will not contain an entry for that terminator.
/// Also populates `suspendOps` with all `llhd.wait` and `llhd.halt` ops.
void DriveHoister::collectDriveSets() {
  SmallPtrSet<Value, 8> unhoistableSlots;
  SmallDenseMap<Value, DriveOp, 8> laterDrives;

  auto trueAttr = BoolAttr::get(processOp.getContext(), true);
  for (auto &block : processOp.getBody()) {
    // We can only hoist drives before wait or halt terminators.
    auto *terminator = block.getTerminator();
    if (!isa<WaitOp, HaltOp>(terminator))
      continue;
    suspendOps.push_back(terminator);

    bool beyondSideEffect = false;
    laterDrives.clear();

    for (auto &op : llvm::make_early_inc_range(
             llvm::reverse(block.without_terminator()))) {
      auto driveOp = dyn_cast<DriveOp>(op);

      // We can only hoist drives that have no side-effecting ops between
      // themselves and the terminator of the block. If we see a side-effecting
      // op, give up on this block.
      if (!driveOp) {
        if (!isMemoryEffectFree(&op))
          beyondSideEffect = true;
        continue;
      }

      // Check if we can hoist drives to this signal.
      if (!slots.contains(driveOp.getSignal()) ||
          unhoistableSlots.contains(driveOp.getSignal())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping (slot unhoistable): " << driveOp << "\n");
        continue;
      }

      // If this drive is beyond a side-effecting op, mark the slot as
      // unhoistable.
      if (beyondSideEffect) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Aborting slot (drive across side-effect): " << driveOp
                   << "\n");
        unhoistableSlots.insert(driveOp.getSignal());
        continue;
      }

      // Handle the case where we've seen a later drive to this slot.
      auto &laterDrive = laterDrives[driveOp.getSignal()];
      if (laterDrive) {
        // If there is a later drive with the same delay and enable condition,
        // we can simply ignore this drive.
        if (laterDrive.getTime() == driveOp.getTime() &&
            laterDrive.getEnable() == driveOp.getEnable()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "- Skipping (driven later): " << driveOp << "\n");
          continue;
        }

        // Otherwise mark the slot as unhoistable since we cannot merge multiple
        // drives with different delays or enable conditions into a single
        // drive.
        LLVM_DEBUG(llvm::dbgs()
                   << "- Aborting slot (multiple drives): " << driveOp << "\n");
        unhoistableSlots.insert(driveOp.getSignal());
        continue;
      }
      laterDrive = driveOp;

      // Add the operands of this drive to the drive set for the driven slot.
      auto operands = DriveOperands{
          driveOp.getValue(),
          driveOp.getTime(),
          driveOp.getEnable() ? DriveValue(driveOp.getEnable())
                              : DriveValue(trueAttr),
      };
      auto &driveSet = driveSets[driveOp.getSignal()];
      driveSet.ops.insert(driveOp);
      driveSet.operands.insert({terminator, operands});
    }
  }

  // Remove slots we've found to be unhoistable.
  slots.remove_if([&](auto slot) {
    if (unhoistableSlots.contains(slot)) {
      driveSets.erase(slot);
      return true;
    }
    return false;
  });
}

/// Make sure all drive sets specify a drive value for each terminator. If a
/// terminator is missing, add a drive with its enable set to false. Also
/// determine which values are uniform and available outside the process, such
/// that we don't create unnecessary process results.
void DriveHoister::finalizeDriveSets() {
  auto falseAttr = BoolAttr::get(processOp.getContext(), false);
  for (auto &[slot, driveSet] : driveSets) {
    // Insert drives with enable set to false for any terminators that are
    // missing. This ensures that the drive set contains information for every
    // terminator.
    for (auto *suspendOp : suspendOps) {
      auto operands = DriveOperands{
          DriveValue::dontCare(cast<RefType>(slot.getType()).getNestedType()),
          DriveValue::dontCare(TimeType::get(processOp.getContext())),
          DriveValue(falseAttr),
      };
      driveSet.operands.insert({suspendOp, operands});
    }

    // Determine which drive operands have a uniform value across all
    // terminators. A null `DriveValue` indicates that there is no uniform
    // value.
    auto unify = [](DriveValue &accumulator, DriveValue other) {
      if (other.isDontCare())
        return;
      if (accumulator == other)
        return;
      accumulator = accumulator.isDontCare() ? other : DriveValue();
    };
    assert(!driveSet.operands.empty());
    driveSet.uniform = driveSet.operands.begin()->second;
    for (auto [terminator, otherOperands] : driveSet.operands) {
      unify(driveSet.uniform.value, otherOperands.value);
      unify(driveSet.uniform.delay, otherOperands.delay);
      unify(driveSet.uniform.enable, otherOperands.enable);
    }

    // Discard uniform non-constant values. We cannot directly use SSA values
    // defined outside the process in the extracted drive, since those values
    // may change at different times than the current process executes and
    // updates its results.
    auto clearIfNonConst = [&](DriveValue &driveValue) {
      if (isa_and_nonnull<Value>(driveValue.data))
        driveValue = DriveValue();
    };
    clearIfNonConst(driveSet.uniform.value);
    clearIfNonConst(driveSet.uniform.delay);
    clearIfNonConst(driveSet.uniform.enable);
  }

  LLVM_DEBUG({
    for (auto slot : slots) {
      const auto &driveSet = driveSets[slot];
      llvm::dbgs() << "Drives to " << slot << "\n";
      if (driveSet.uniform.value)
        llvm::dbgs() << "- Uniform value: " << driveSet.uniform.value << "\n";
      if (driveSet.uniform.delay)
        llvm::dbgs() << "- Uniform delay: " << driveSet.uniform.delay << "\n";
      if (driveSet.uniform.enable)
        llvm::dbgs() << "- Uniform enable: " << driveSet.uniform.enable << "\n";
      for (auto *suspendOp : suspendOps) {
        auto operands = driveSet.operands.lookup(suspendOp);
        llvm::dbgs() << "- At " << *suspendOp << "\n";
        if (!driveSet.uniform.value)
          llvm::dbgs() << "  - Value: " << operands.value << "\n";
        if (!driveSet.uniform.delay)
          llvm::dbgs() << "  - Delay: " << operands.delay << "\n";
        if (!driveSet.uniform.enable)
          llvm::dbgs() << "  - Enable: " << operands.enable << "\n";
      }
    }
  });
}

/// Hoist drive operations out of the process. This function adds yield operands
/// to carry the operands of the hoisted drives out of the process, and adds
/// corresponding process results. It then creates replacement drives outside of
/// the process and uses the new result values for the drive operands.
void DriveHoister::hoistDrives() {
  if (driveSets.empty())
    return;
  LLVM_DEBUG(llvm::dbgs() << "Hoisting drives of " << driveSets.size()
                          << " slots\n");

  // A builder to construct constant values outside the region.
  OpBuilder builder(processOp);
  SmallDenseMap<Attribute, Value> materializedConstants;

  auto materialize = [&](DriveValue driveValue) -> Value {
    OpBuilder builder(processOp);
    return TypeSwitch<DriveValue::Data, Value>(driveValue.data)
        .Case<Value>([](auto value) { return value; })
        .Case<IntegerAttr>([&](auto attr) {
          auto &slot = materializedConstants[attr];
          if (!slot)
            slot = hw::ConstantOp::create(builder, processOp.getLoc(), attr);
          return slot;
        })
        .Case<TimeAttr>([&](auto attr) {
          auto &slot = materializedConstants[attr];
          if (!slot)
            slot = ConstantTimeOp::create(builder, processOp.getLoc(), attr);
          return slot;
        })
        .Case<Type>([&](auto type) {
          // TODO: This should probably create something like a `llhd.dontcare`.
          if (isa<TimeType>(type)) {
            auto attr = TimeAttr::get(builder.getContext(), 0, "ns", 0, 0);
            auto &slot = materializedConstants[attr];
            if (!slot)
              slot = ConstantTimeOp::create(builder, processOp.getLoc(), attr);
            return slot;
          }
          auto numBits = hw::getBitWidth(type);
          assert(numBits >= 0);
          Value value = hw::ConstantOp::create(
              builder, processOp.getLoc(), builder.getIntegerType(numBits), 0);
          if (value.getType() != type)
            value =
                hw::BitcastOp::create(builder, processOp.getLoc(), type, value);
          return value;
        });
  };

  // Add the non-uniform drive operands as yield operands of any `llhd.wait` and
  // `llhd.halt` terminators.
  for (auto *suspendOp : suspendOps) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Adding yield operands to " << *suspendOp << "\n");
    MutableOperandRange yieldOperands =
        TypeSwitch<Operation *, MutableOperandRange>(suspendOp)
            .Case<WaitOp, HaltOp>(
                [](auto op) { return op.getYieldOperandsMutable(); });

    auto addYieldOperand = [&](DriveValue uniform, DriveValue nonUniform) {
      if (!uniform)
        yieldOperands.append(materialize(nonUniform));
    };

    for (auto slot : slots) {
      auto &driveSet = driveSets[slot];
      auto operands = driveSet.operands.lookup(suspendOp);
      addYieldOperand(driveSet.uniform.value, operands.value);
      addYieldOperand(driveSet.uniform.delay, operands.delay);
      addYieldOperand(driveSet.uniform.enable, operands.enable);
    }
  }

  // Add process results corresponding to the added yield operands.
  SmallVector<Type> resultTypes(processOp->getResultTypes());
  auto oldNumResults = resultTypes.size();
  auto addResultType = [&](DriveValue uniform, DriveValue nonUniform) {
    if (!uniform)
      resultTypes.push_back(nonUniform.getType());
  };
  for (auto slot : slots) {
    auto &driveSet = driveSets[slot];
    auto operands = driveSet.operands.begin()->second;
    addResultType(driveSet.uniform.value, operands.value);
    addResultType(driveSet.uniform.delay, operands.delay);
    addResultType(driveSet.uniform.enable, operands.enable);
  }
  auto newProcessOp =
      ProcessOp::create(builder, processOp.getLoc(), resultTypes,
                        processOp->getOperands(), processOp->getAttrs());
  newProcessOp.getBody().takeBody(processOp.getBody());
  processOp.replaceAllUsesWith(
      newProcessOp->getResults().slice(0, oldNumResults));
  processOp.erase();
  processOp = newProcessOp;

  // Hoist the actual drive operations. We either materialize uniform values
  // directly, since they are guaranteed to be able outside the process at this
  // point, or use the new process results.
  builder.setInsertionPointAfter(processOp);
  auto newResultIdx = oldNumResults;

  auto useResultValue = [&](DriveValue uniform) {
    if (!uniform)
      return processOp.getResult(newResultIdx++);
    return materialize(uniform);
  };

  auto removeIfUnused = [](Value value) {
    if (value)
      if (auto *defOp = value.getDefiningOp())
        if (defOp && isOpTriviallyDead(defOp))
          defOp->erase();
  };

  auto trueAttr = builder.getBoolAttr(true);
  for (auto slot : slots) {
    auto &driveSet = driveSets[slot];

    // Create the new drive outside of the process.
    auto value = useResultValue(driveSet.uniform.value);
    auto delay = useResultValue(driveSet.uniform.delay);
    auto enable = driveSet.uniform.enable != DriveValue(trueAttr)
                      ? useResultValue(driveSet.uniform.enable)
                      : Value{};
    [[maybe_unused]] auto newDrive =
        DriveOp::create(builder, slot.getLoc(), slot, value, delay, enable);
    LLVM_DEBUG(llvm::dbgs() << "- Add " << newDrive << "\n");

    // Remove the old drives inside of the process.
    for (auto *oldOp : driveSet.ops) {
      auto oldDrive = cast<DriveOp>(oldOp);
      LLVM_DEBUG(llvm::dbgs() << "- Remove " << oldDrive << "\n");
      auto delay = oldDrive.getTime();
      auto enable = oldDrive.getEnable();
      oldDrive.erase();
      removeIfUnused(delay);
      removeIfUnused(enable);
    }
    driveSet.ops.clear();
  }
  assert(newResultIdx == processOp->getNumResults());
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct HoistSignalsPass
    : public llhd::impl::HoistSignalsPassBase<HoistSignalsPass> {
  void runOnOperation() override;
};
} // namespace

void HoistSignalsPass::runOnOperation() {
  SmallVector<Region *> regions;
  getOperation()->walk([&](Operation *op) {
    if (isa<ProcessOp, FinalOp, CombinationalOp, scf::IfOp>(op))
      for (auto &region : op->getRegions())
        if (!region.empty())
          regions.push_back(&region);
  });
  for (auto *region : regions) {
    ProbeHoister(*region).hoist();
    if (auto processOp = dyn_cast<ProcessOp>(region->getParentOp()))
      DriveHoister(processOp).hoist();
  }
}
