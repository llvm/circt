//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DNF.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#define DEBUG_TYPE "llhd-deseq"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_DESEQPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallMapVector;
using llvm::SmallSetVector;

namespace circt {
namespace llhd {
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DNF &dnf) {
  dnf.printWithValues(os);
  return os;
}
} // namespace llhd
} // namespace circt

//===----------------------------------------------------------------------===//
// Value Table
//===----------------------------------------------------------------------===//

namespace {

/// An entry in a value table. Associates the value assigned to an SSA value
/// with a condition under which the association holds.
struct ValueEntry {
  DNF condition;
  Value value;

  int compare(const ValueEntry &other) const {
    auto order = condition.compare(other.condition);
    if (order != 0)
      return order;
    if (value.getAsOpaquePointer() < other.value.getAsOpaquePointer())
      return -1;
    if (value.getAsOpaquePointer() > other.value.getAsOpaquePointer())
      return 1;
    return 0;
  }

  bool operator<(const ValueEntry &other) const { return compare(other) < 0; }
};

/// A table of values and the conditions under which the values apply. This
/// struct is used to track the different concrete values an SSA value may have.
/// Block arguments receive different concrete values from their predecessors,
/// which this table can track separately given the condition under which
/// control reaches a block from its predecessors.
struct ValueTable {
  /// The entries in the table.
  SmallVector<ValueEntry, 1> entries;

  /// Create an empty value table.
  ValueTable() {}
  /// Create a table with a single value produced under all conditions.
  explicit ValueTable(Value value) : ValueTable(DNF(true), value) {}
  /// Create a table with a single value produced under a give ncondition.
  ValueTable(DNF condition, Value value) {
    if (!condition.isFalse())
      entries.push_back(ValueEntry{condition, value});
  }

  /// Check whether this table is empty.
  bool isEmpty() const { return entries.empty(); }

  /// Compare this table to another.
  int compare(const ValueTable &other) const;
  bool operator==(const ValueTable &other) const { return compare(other) == 0; }
  bool operator!=(const ValueTable &other) const { return compare(other) != 0; }
  bool operator<(const ValueTable &other) const { return compare(other) < 0; }
  bool operator>(const ValueTable &other) const { return compare(other) > 0; }
  bool operator>=(const ValueTable &other) const { return compare(other) >= 0; }
  bool operator<=(const ValueTable &other) const { return compare(other) <= 0; }

  /// Merge the values of another table into this table.
  void merge(ValueTable &&other);
  /// Add a condition to all entries in this table. Erases entries from the
  /// table that become trivially false.
  void addCondition(const DNF &condition);
  /// Create a reduced table that only contains values which overlap with a give
  /// ncondition.
  ValueTable filtered(const DNF &condition);
};

} // namespace

int ValueTable::compare(const ValueTable &other) const {
  if (entries.size() < other.entries.size())
    return -1;
  if (entries.size() > other.entries.size())
    return 1;
  for (auto [thisEntry, otherEntry] : llvm::zip(entries, other.entries)) {
    auto order = thisEntry.compare(otherEntry);
    if (order != 0)
      return order;
  }
  return 0;
}

void ValueTable::merge(ValueTable &&other) {
  SmallDenseMap<Value, unsigned> seenValues;
  for (auto [index, entry] : llvm::enumerate(entries))
    seenValues[entry.value] = index;
  for (auto &&entry : other.entries) {
    if (auto it = seenValues.find(entry.value); it != seenValues.end()) {
      entries[it->second].condition |= entry.condition;
    } else {
      seenValues.insert({entry.value, entries.size()});
      entries.push_back(entry);
    }
  }
  llvm::sort(entries);
}

void ValueTable::addCondition(const DNF &condition) {
  llvm::erase_if(entries, [&](auto &entry) {
    entry.condition &= condition;
    return entry.condition.isFalse();
  });
}

ValueTable ValueTable::filtered(const DNF &condition) {
  ValueTable result;
  for (auto entry : entries) {
    entry.condition &= condition;
    if (entry.condition.isFalse())
      continue;

    // Remove the AND terms that are implied by the condition.
    for (auto &orTerm : entry.condition.orTerms)
      llvm::erase_if(orTerm.andTerms, [&](auto andTerm) {
        return llvm::any_of(condition.orTerms, [&](auto &conditionOrTerm) {
          return llvm::is_contained(conditionOrTerm.andTerms, andTerm);
        });
      });

    result.entries.push_back(std::move(entry));
  }
  return result;
}

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ValueTable &table) {
  SmallMapVector<Value, unsigned, 8> valueIds;
  auto dumpValue = [&](Value value) {
    auto id = valueIds.insert({value, valueIds.size()}).first->second;
    os << "v" << id;
  };

  // Print the table itself.
  os << "ValueTable(";
  llvm::interleaveComma(table.entries, os, [&](auto &entry) {
    entry.condition.print(os, dumpValue);
    os << " -> ";
    Attribute attr;
    if (auto *defOp = entry.value.getDefiningOp();
        defOp && defOp->template hasTrait<OpTrait::ConstantLike>() &&
        m_Constant(&attr).match(defOp))
      os << attr;
    else
      dumpValue(entry.value);
  });

  // Print the values used.
  if (!valueIds.empty()) {
    os << " with ";
    llvm::interleaveComma(valueIds, os, [&](auto valueAndId) {
      os << "v" << valueAndId.second << " = ";
      valueAndId.first.printAsOperand(os, OpPrintingFlags());
    });
  }
  os << ")";

  return os;
}

//===----------------------------------------------------------------------===//
// Block Successor Values
//===----------------------------------------------------------------------===//

namespace {

/// A struct that represents a single control flow edge from a predecessor block
/// to a one of its successors. It tracks the condition under which control
/// transfers along this edge, which may be `true` in case of a `cf.br`, or a
/// concrete branch condition for `cf.cond_br`. Also tracks any DNF expression
/// and value table associated with each block argument of the successor block.
struct SuccessorValue {
  /// The condition under which control transfers to this successor.
  DNF condition;
  /// The DNF expression for each successor block argument. Null for any non-i1
  /// arguments.
  SmallVector<DNF, 1> booleanArgs;
  /// The value table for each successor block argument.
  SmallVector<ValueTable, 1> valueArgs;

  bool operator==(const SuccessorValue &other) const {
    return condition == other.condition && booleanArgs == other.booleanArgs &&
           valueArgs == other.valueArgs;
  }

  bool operator!=(const SuccessorValue &other) const {
    return !(*this == other);
  }
};

/// A list of conditions and block argument values transferred along control
/// flow edges from a predecessor to each of its successors. Contains an entry
/// for each successor.
using SuccessorValues = SmallVector<SuccessorValue, 2>;

} // namespace

//===----------------------------------------------------------------------===//
// Clock and Reset Analysis
//===----------------------------------------------------------------------===//

namespace {

/// A single reset extracted from a process during trigger analysis.
struct ResetInfo {
  /// The value acting as the reset, causing the register to be set to `value`
  /// when triggered.
  Value reset;
  /// The value the register is reset to.
  Value value;
  /// Whether the reset is active when high.
  bool activeHigh;

  /// Check if this reset info is null.
  operator bool() const { return bool(reset); }
};

/// An edge on a trigger.
enum class Edge { None = 0, Pos, Neg };

/// A single clock extracted from a process during trigger analysis.
struct ClockInfo {
  /// The value acting as the clock, causing the register to be set to a value
  /// in `valueTable` when triggered.
  Value clock;
  /// The edge which causes the update. Guaranteed to be `Pos` or `Neg`.
  Edge edge;
  /// The table of values that the register will assume when triggered by the
  /// clock. This may contain a single unconditional value, or may list a
  /// concrete list of values which are assumed under certain conditions. This
  /// can be used to infer the use of a register with or without an enable line.
  ValueTable valueTable;

  /// Check if this clock info is null.
  operator bool() const { return bool(clock); }
};

/// A drive op and the clock and reset that resulted from trigger analysis. A
/// process may describe multiple clock and reset triggers, but since the
/// registers we lower to only allow a single clock and a single reset, this
/// struct tracks a single clock and reset, respectively. Processes describing
/// multiple clocks or resets are skipped.
struct DriveInfo {
  /// The drive operation.
  DrvOp op;
  /// The clock that triggers a change to the driven value. Guaranteed to be
  /// non-null.
  ClockInfo clock;
  /// The optional reset that triggers a change of the driven value to a fixed
  /// reset value. Null if no reset was detected.
  ResetInfo reset;

  DriveInfo() {}
  explicit DriveInfo(DrvOp op) : op(op) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// Value Assignments for Process Specialization
//===----------------------------------------------------------------------===//

namespace {

/// A single `i1` value that is fixed to a given value in the past and the
/// present.
struct FixedValue {
  /// The IR value being fixed.
  Value value;
  /// The assigned value in the past, as transported into the presented via a
  /// destination operand of a process' wait op.
  bool past;
  /// The assigned value in the present.
  bool present;

  bool operator==(const FixedValue &other) const {
    return value == other.value && past == other.past &&
           present == other.present;
  }

  operator bool() const { return bool(value); }
};

/// A list of `i1` values that are fixed to a given value. These are used when
/// specializing a process to compute the value and enable condition for a drive
/// when a trigger occurs.
using FixedValues = SmallVector<FixedValue, 2>;

llvm::hash_code hash_value(const FixedValue &arg) {
  return llvm::hash_combine(arg.value, arg.past, arg.present);
}

llvm::hash_code hash_value(const FixedValues &arg) {
  return llvm::hash_combine_range(arg.begin(), arg.end());
}

} // namespace

// Allow `FixedValue` and `FixedValues` to be used as hash map keys.
namespace llvm {
template <>
struct DenseMapInfo<FixedValue> {
  static inline FixedValue getEmptyKey() {
    return FixedValue{DenseMapInfo<Value>::getEmptyKey(), false, false};
  }
  static inline FixedValue getTombstoneKey() {
    return FixedValue{DenseMapInfo<Value>::getTombstoneKey(), false, false};
  }
  static unsigned getHashValue(const FixedValue &key) {
    return hash_value(key);
  }
  static bool isEqual(const FixedValue &a, const FixedValue &b) {
    return a == b;
  }
};
template <>
struct DenseMapInfo<FixedValues> {
  static inline FixedValues getEmptyKey() {
    return {DenseMapInfo<FixedValue>::getEmptyKey()};
  }
  static inline FixedValues getTombstoneKey() {
    return {DenseMapInfo<FixedValue>::getTombstoneKey()};
  }
  static unsigned getHashValue(const FixedValues &key) {
    return hash_value(key);
  }
  static bool isEqual(const FixedValues &a, const FixedValues &b) {
    return a == b;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Desequentialization
//===----------------------------------------------------------------------===//

namespace {
/// The work horse promoting processes into concrete registers.
struct Deseq {
  Deseq(ProcessOp process) : process(process) {}
  void deseq();

  bool analyzeProcess();
  bool analyzeTriggers();
  bool matchDrives();
  bool matchDrive(DriveInfo &drive);
  bool isValidResetValue(Value value);
  void implementRegisters();
  void implementRegister(DriveInfo &drive);
  Value specializeValue(Value value, FixedValues fixedValues);
  ValueRange specializeProcess(FixedValues fixedValues);

  void updateBoolean(Value value, DNF dnf);
  void updateValue(Value value, ValueTable table);
  void updateSuccessors(Block *block, SuccessorValues values);
  void updateCondition(Block *block, DNF condition);

  void markDirty(Operation *op);
  void markDirty(Block *block);

  void propagate();
  void propagate(Block *block);
  void propagate(Operation *op);
  void propagate(cf::BranchOp op);
  void propagate(cf::CondBranchOp op);
  void propagate(WaitOp op);
  void propagate(comb::OrOp op);
  void propagate(comb::AndOp op);
  void propagate(comb::XorOp op);
  void propagate(comb::MuxOp op);

  /// The process we are desequentializing.
  ProcessOp process;
  /// The single wait op of the process.
  WaitOp wait;
  /// The boolean values observed by the wait.
  SmallSetVector<Value, 2> observedBooleans;
  /// The conditional drive operations fed by this process.
  SmallVector<DriveInfo> driveInfos;
  /// The triggers that cause the process to update its results.
  SmallSetVector<Value, 2> triggers;
  /// The values carried from the past into the present as destination operands
  /// of the wait op. These values are guaranteed to also be contained in
  /// `triggers` and `observedBooleans`.
  SmallVector<Value, 2> pastValues;
  /// Specializations of the process for different trigger values.
  SmallDenseMap<FixedValues, ValueRange, 2> specializedProcesses;
  /// An `llhd.constant_time` op created to represent an epsilon delay.
  ConstantTimeOp epsilonDelay;
  /// A map of operations that have been checked to be valid reset values.
  DenseMap<Operation *, bool> staticOps;

  /// A worklist of nodes to be updated during the data flow analysis.
  SmallSetVector<PointerUnion<Operation *, Block *>, 4> dirtyNodes;
  /// The DNF expression computed for an `i1` value in the IR.
  DenseMap<Value, DNF> booleanLattice;
  /// The value table computed for a value in the IR. This essentially lists
  /// what values an SSA value assumes under certain conditions.
  DenseMap<Value, ValueTable> valueLattice;
  /// The condition under which control flow reaches a block. The block
  /// immediately following the wait op has this set to true; any further
  /// conditional branches will refine the condition of successor blocks.
  DenseMap<Block *, DNF> blockConditionLattice;
  /// The conditions and values transferred from a block to its successors.
  DenseMap<Block *, SuccessorValues> successorLattice;
};
} // namespace

/// Try to lower the process to a set of registers.
void Deseq::deseq() {
  // Check whether the process meets the basic criteria for being replaced by a
  // register. This includes having only a single `llhd.wait` op and feeding
  // particular kinds of `llhd.drv` ops.
  if (!analyzeProcess())
    return;
  LLVM_DEBUG({
    llvm::dbgs() << "Desequentializing " << process.getLoc() << "\n";
    llvm::dbgs() << "- Feeds " << driveInfos.size() << " conditional drives\n";
  });

  // Perform a data flow analysis to find SSA values corresponding to a detected
  // rising or falling edge, and which values are driven under which conditions.
  propagate();

  // Check which values are used as triggers.
  if (!analyzeTriggers())
    return;
  LLVM_DEBUG({
    llvm::dbgs() << "- " << triggers.size() << " potential triggers:\n";
    for (auto trigger : triggers)
      llvm::dbgs() << "  - " << trigger << "\n";
  });

  // For each drive fed by this process determine the exact triggers that cause
  // them to drive a new value, and ensure that the behavior can be represented
  // by a register.
  if (!matchDrives())
    return;

  // Make the drives unconditional and capture the conditional behavior as
  // register operations.
  implementRegisters();

  // At this point the process has been replaced with specialized versions of it
  // for the different triggers and can be removed.
  process.erase();
}

//===----------------------------------------------------------------------===//
// Process Analysis
//===----------------------------------------------------------------------===//

/// Determine whether we can desequentialize the current process. Also gather
/// the wait and drive ops that are relevant.
bool Deseq::analyzeProcess() {
  // We can only desequentialize processes with no side-effecting ops.
  if (!isMemoryEffectFree(process)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping " << process.getLoc() << ": has side effects\n");
    return false;
  }

  // Find the single wait op.
  for (auto &block : process.getBody()) {
    if (auto candidate = dyn_cast<WaitOp>(block.getTerminator())) {
      if (wait) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                                << ": has multiple waits\n");
        return false;
      }
      wait = candidate;
    }
  }
  if (!wait) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping " << process.getLoc() << ": has no wait\n");
    return false;
  }
  for (auto value : wait.getObserved())
    if (value.getType().isSignlessInteger(1))
      observedBooleans.insert(value);

  // Ensure that all process results lead to conditional drive operations.
  SmallPtrSet<Operation *, 8> seenDrives;
  for (auto &use : process->getUses()) {
    auto driveOp = dyn_cast<DrvOp>(use.getOwner());
    if (!driveOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping " << process.getLoc() << ": feeds non-drive "
                 << use.getOwner()->getLoc() << "\n");
      return false;
    }
    if (!seenDrives.insert(driveOp).second)
      continue;

    // We can only deal with conditional drives.
    if (!driveOp.getEnable()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping " << process.getLoc()
                 << ": feeds unconditional drive " << driveOp << "\n");
      return false;
    }

    // We can only deal with the process result being used as drive value or
    // condition.
    if (use.getOperandNumber() != 1 && use.getOperandNumber() != 2) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping " << process.getLoc()
                 << ": feeds drive operand that is neither value nor enable: "
                 << driveOp << "\n");
      return false;
    }

    driveInfos.push_back(DriveInfo(driveOp));
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Data Flow Analysis
//===----------------------------------------------------------------------===//

/// Add a block to the data flow analysis worklist.
void Deseq::markDirty(Block *block) { dirtyNodes.insert(block); }

/// Add an operation to the data flow analysis worklist.
void Deseq::markDirty(Operation *op) {
  if (op->getParentRegion()->isAncestor(&process.getBody()) ||
      process.getBody().isAncestor(op->getParentRegion()))
    dirtyNodes.insert(op);
}

/// Update the boolean lattice value assigned to a value in the IR, and mark all
/// dependent nodes dirty.
void Deseq::updateBoolean(Value value, DNF dnf) {
  assert(value.getType().isSignlessInteger(1) && "can only trace i1 DNFs");
  auto &slot = booleanLattice[value];
  if (slot != dnf) {
    LLVM_DEBUG({
      llvm::dbgs() << "- Update ";
      value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " = " << dnf << "\n";
    });
    slot = dnf;
    for (auto *user : value.getUsers())
      markDirty(user);
  }
}

/// Update the general lattice value assigned to a value in the IR, and mark all
/// dependent nodes dirty.
void Deseq::updateValue(Value value, ValueTable table) {
  auto &slot = valueLattice[value];
  if (slot != table) {
    LLVM_DEBUG({
      llvm::dbgs() << "- Update ";
      value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " = " << table << "\n";
    });
    slot = table;
    for (auto *user : value.getUsers())
      markDirty(user);
  }
}

/// Update the successor lattice values of a block, and mark all dependent nodes
/// dirty.
void Deseq::updateSuccessors(Block *block, SuccessorValues values) {
  auto &slot = successorLattice[block];
  for (auto [index, successor] : llvm::enumerate(block->getSuccessors())) {
    if (!slot.empty() && slot[index] == values[index])
      continue;
    markDirty(successor);
  }
  slot = values;
}

/// Update the condition lattice value of a block, and mark all dependent nodes
/// as dirty.
void Deseq::updateCondition(Block *block, DNF condition) {
  auto &slot = blockConditionLattice[block];
  if (slot != condition) {
    LLVM_DEBUG({
      llvm::dbgs() << "- Update ";
      block->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " condition = " << condition << "\n";
    });
    slot = condition;
    markDirty(block->getTerminator());
  }
}

/// Perform a data flow analysis of the values observed by the process' wait op,
/// propagated from the past into the present as destination operands of the
/// wait op, and values yielded as process results. This analysis distills out
/// drive conditions as canonical DNF expressions and determines under which
/// conditions given values are produced as process results. This information is
/// then used to detect if a process expresses a pos/neg edge trigger detection,
/// among other things.
void Deseq::propagate() {
  // Seed the lattice for any value defined outside the process.
  SmallPtrSet<Operation *, 8> opsAboveSeen;
  mlir::visitUsedValuesDefinedAbove(
      process->getRegions(), [&](auto *opOperand) {
        auto value = opOperand->get();
        if (auto *defOp = value.getDefiningOp();
            defOp && !observedBooleans.contains(value)) {
          if (opsAboveSeen.insert(defOp).second)
            propagate(defOp);
        } else {
          if (value.getType().isSignlessInteger(1))
            updateBoolean(value, DNF(value));
          updateValue(value, ValueTable(value));
        }
      });

  // Seed the lattice for all operations in the process body.
  for (auto &block : process.getBody()) {
    propagate(&block);
    for (auto &op : block)
      propagate(&op);
  }

  // Propagate lattice values.
  while (!dirtyNodes.empty()) {
    auto node = dirtyNodes.pop_back_val();
    if (auto *op = dyn_cast<Operation *>(node))
      propagate(op);
    else
      propagate(cast<Block *>(node));
  }
  LLVM_DEBUG(llvm::dbgs() << "- Finished propagation\n");
}

/// Combine the control flow conditions and block argument values from
/// predecessor blocks into a unified condition and values for each of the
/// block's arguments.
void Deseq::propagate(Block *block) {
  // Combine all the values coming from our predecessor blocks.
  const unsigned numArgs = block->getNumArguments();
  auto condition = DNF(false);
  SmallVector<DNF> booleanArgs(numArgs, DNF(false));
  SmallVector<ValueTable> valueArgs(numArgs, ValueTable());

  SmallPtrSet<Block *, 4> seen;
  for (auto *predecessor : block->getPredecessors()) {
    if (!seen.insert(predecessor).second)
      continue;
    auto &successorValues = successorLattice[predecessor];
    if (successorValues.empty())
      continue;
    auto *terminator = predecessor->getTerminator();
    for (auto &blockOperand : terminator->getBlockOperands()) {
      if (blockOperand.get() != block)
        continue;
      auto &successorValue = successorValues[blockOperand.getOperandNumber()];
      condition |= successorValue.condition;
      for (unsigned argIdx = 0; argIdx < numArgs; ++argIdx) {
        booleanArgs[argIdx] |=
            successorValue.booleanArgs[argIdx] & successorValue.condition;
        auto valueTable = successorValue.valueArgs[argIdx];
        valueTable.addCondition(successorValue.condition);
        valueArgs[argIdx].merge(std::move(valueTable));
      }
    }
  }

  // Update the block condition.
  updateCondition(block, condition);

  // Update the individual arguments.
  for (auto [arg, booleanArg, valueArg] :
       llvm::zip(block->getArguments(), booleanArgs, valueArgs)) {
    if (arg.getType().isSignlessInteger(1))
      updateBoolean(arg, booleanArg);
    updateValue(arg, valueArg);
  }
}

/// Propagate lattice values across an operation.
void Deseq::propagate(Operation *op) {
  // Handle boolean constants.
  if (op->hasTrait<OpTrait::ConstantLike>() && op->getNumResults() == 1) {
    APInt intValue;
    if (op->getResult(0).getType().isSignlessInteger(1) &&
        m_ConstantInt(&intValue).match(op)) {
      assert(intValue.getBitWidth() == 1);
      updateBoolean(op->getResult(0), DNF(intValue.isOne()));
    }
    updateValue(op->getResult(0), ValueTable(op->getResult(0)));
    return;
  }

  // Handle all other ops.
  TypeSwitch<Operation *>(op)
      .Case<cf::BranchOp, cf::CondBranchOp, WaitOp, comb::OrOp, comb::AndOp,
            comb::XorOp, comb::MuxOp>([&](auto op) {
        // Handle known ops.
        propagate(op);
      })
      .Default([&](auto) {
        // All other ops will simply produce their results as opaque values.
        for (auto result : op->getResults()) {
          if (result.getType().isSignlessInteger(1))
            updateBoolean(result, DNF(result));
          updateValue(result, ValueTable(result));
        }
      });
}

/// Propagate lattice values across an unconditional branch.
void Deseq::propagate(cf::BranchOp op) {
  SuccessorValue latticeValue;
  latticeValue.condition = blockConditionLattice[op->getBlock()];
  latticeValue.booleanArgs.reserve(op.getDestOperands().size());
  latticeValue.valueArgs.reserve(op.getDestOperands().size());
  for (auto operand : op.getDestOperands()) {
    latticeValue.booleanArgs.push_back(booleanLattice.lookup(operand));
    latticeValue.valueArgs.push_back(valueLattice.lookup(operand));
  }
  updateSuccessors(op->getBlock(), {latticeValue});
}

/// Propagate lattice values across a conditional branch. This combines the
/// parent block's condition of control flow reaching it with the branch
/// condition to determine the condition under which the successor blocks will
/// be reached.
void Deseq::propagate(cf::CondBranchOp op) {
  SuccessorValues latticeValues(2, SuccessorValue());
  auto blockCondition = blockConditionLattice[op->getBlock()];
  auto branchCondition = booleanLattice.lookup(op.getCondition());

  // Handle the true branch.
  auto &trueValue = latticeValues[0];
  trueValue.condition = blockCondition & branchCondition;
  trueValue.booleanArgs.reserve(op.getTrueDestOperands().size());
  trueValue.valueArgs.reserve(op.getTrueDestOperands().size());
  for (auto operand : op.getTrueDestOperands()) {
    trueValue.booleanArgs.push_back(booleanLattice.lookup(operand));
    trueValue.valueArgs.push_back(valueLattice.lookup(operand));
  }

  // Handle the false branch.
  auto &falseValue = latticeValues[1];
  falseValue.condition = blockCondition & ~branchCondition;
  falseValue.booleanArgs.reserve(op.getFalseDestOperands().size());
  falseValue.valueArgs.reserve(op.getFalseDestOperands().size());
  for (auto operand : op.getFalseDestOperands()) {
    falseValue.booleanArgs.push_back(booleanLattice.lookup(operand));
    falseValue.valueArgs.push_back(valueLattice.lookup(operand));
  }

  updateSuccessors(op->getBlock(), latticeValues);
}

/// Propagate lattice values across a wait op. This will mark values flowing
/// into the destination block as past values, such that the data flow analysis
/// can determine if a value is compared against a past version of itself. Also
/// updates the parent process' result values based on the wait's yield
/// operands.
void Deseq::propagate(WaitOp op) {
  SuccessorValue latticeValue;
  latticeValue.condition = DNF(true); // execution resumes in the destination
  latticeValue.booleanArgs.reserve(op.getDestOperands().size());
  latticeValue.valueArgs.reserve(op.getDestOperands().size());
  for (auto operand : op.getDestOperands()) {
    // Destination operands of the wait op are essentially carrying a past value
    // into the destination block.
    auto value = booleanLattice.lookup(operand);
    if (value) {
      bool nonePastAlready = llvm::all_of(value.orTerms, [](auto &orTerm) {
        return llvm::all_of(orTerm.andTerms, [](auto &andTerm) {
          if (andTerm.hasUse(AndTerm::Past) || andTerm.hasUse(AndTerm::NotPast))
            return false;
          andTerm.uses <<= 1; // map Id/NotId to Past/NotPast
          return true;
        });
      });
      if (!nonePastAlready)
        value = DNF();
    }
    latticeValue.booleanArgs.push_back(value);
    latticeValue.valueArgs.push_back(ValueTable());
  }
  updateSuccessors(op->getBlock(), {latticeValue});

  // If this is the single wait in the current process, update the process
  // results.
  if (op != wait)
    return;
  for (auto [result, operand] :
       llvm::zip(process->getResults(), op.getYieldOperands())) {
    if (operand.getType().isSignlessInteger(1))
      updateBoolean(result, booleanLattice.lookup(operand));
    updateValue(result, valueLattice.lookup(operand));
  }
}

/// Propagate lattice values across a `comb.or` op.
void Deseq::propagate(comb::OrOp op) {
  if (op.getType().isSignlessInteger(1)) {
    auto result = DNF(false);
    for (auto operand : op.getInputs()) {
      result |= booleanLattice.lookup(operand);
      if (result.isTrue())
        break;
    }
    updateBoolean(op, result);
  }
  updateValue(op, ValueTable(op.getResult()));
}

/// Propagate lattice values across a `comb.and` op.
void Deseq::propagate(comb::AndOp op) {
  if (op.getType().isSignlessInteger(1)) {
    auto result = DNF(true);
    for (auto operand : op.getInputs()) {
      result &= booleanLattice.lookup(operand);
      if (result.isFalse())
        break;
    }
    updateBoolean(op, result);
  }
  updateValue(op, ValueTable(op.getResult()));
}

/// Propagate lattice values across a `comb.xor` op.
void Deseq::propagate(comb::XorOp op) {
  if (op.getType().isSignlessInteger(1)) {
    auto result = DNF(false);
    for (auto operand : op.getInputs())
      result ^= booleanLattice.lookup(operand);
    updateBoolean(op, result);
  }
  updateValue(op, ValueTable(op.getResult()));
}

/// Propagate lattice values across a `comb.mux` op.
void Deseq::propagate(comb::MuxOp op) {
  auto condition = booleanLattice.lookup(op.getCond());
  if (op.getType().isSignlessInteger(1)) {
    auto trueValue = booleanLattice.lookup(op.getTrueValue());
    auto falseValue = booleanLattice.lookup(op.getFalseValue());
    updateBoolean(op, condition & trueValue | ~condition & falseValue);
  }
  auto trueValue = valueLattice.lookup(op.getTrueValue());
  auto falseValue = valueLattice.lookup(op.getFalseValue());
  trueValue.addCondition(condition);
  falseValue.addCondition(~condition);
  trueValue.merge(std::move(falseValue));
  updateValue(op, trueValue);
}

//===----------------------------------------------------------------------===//
// Trigger Analysis
//===----------------------------------------------------------------------===//

/// After propagating values across the lattice, determine which values may be
/// involved in trigger detection.
bool Deseq::analyzeTriggers() {
  for (auto operand : wait.getDestOperands()) {
    // Only i1 values can be triggers.
    if (!operand.getType().isSignlessInteger(1)) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Aborting: " << operand.getType() << " trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << "\n";
      });
      return false;
    }

    // Data flow analysis must have produced a single value for this past
    // operand.
    auto dnf = booleanLattice.lookup(operand);
    Value value;
    AndTerm::Use use;
    if (auto single = dnf.getSingleTerm()) {
      std::tie(value, use) = *single;
    } else {
      LLVM_DEBUG({
        llvm::dbgs() << "- Aborting: trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << " with multiple values " << dnf << "\n";
      });
      return false;
    }
    if (use != AndTerm::Id) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Aborting: trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << " is inverted: " << dnf << "\n";
      });
      return false;
    }

    // We can only reason about past values defined outside the process.
    if (process.getBody().isAncestor(value.getParentRegion())) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Aborting: trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << " defined inside the process\n";
      });
      return false;
    }

    // We can only reason about past values if they cause the process to resume
    // when they change.
    if (!observedBooleans.contains(value)) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Aborting: unobserved trigger ";
        value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << "\n";
      });
      return false;
    }

    triggers.insert(value);
    pastValues.push_back(value);
  }

  if (triggers.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "- Aborting: no triggers\n");
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Drive-to-Register Matching
//===----------------------------------------------------------------------===//

/// Match the drives fed by the process against concrete implementable register
/// behaviors. Returns false if any of the drives cannot be implemented as a
/// register.
bool Deseq::matchDrives() {
  for (auto &drive : driveInfos)
    if (!matchDrive(drive))
      return false;
  return true;
}

/// For a given drive op, determine if its drive condition and driven value as
/// determined  by the data flow analysis is implementable by a register op.
/// This function first analyzes the drive condition and extracts concrete
/// posedge or negedge triggers from it. It then analyzes the value driven under
/// each condition and ensures that there is a clear precedence between
/// triggers, for example, to disambiguate a clock from an overriding reset.
/// This function then distills out a single clock and a single optional reset
/// for the drive and stores the information in the given `DriveInfo`. Returns
/// false if no single clock and no single optional reset could be extracted.
bool Deseq::matchDrive(DriveInfo &drive) {
  LLVM_DEBUG(llvm::dbgs() << "- Analyzing " << drive.op << "\n");

  // Determine under which condition the drive is enabled.
  auto condition = booleanLattice.lookup(drive.op.getEnable());
  if (condition.isNull()) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Aborting: null condition on " << drive.op << "\n");
    return false;
  }

  // Determine which value is driven under which conditions.
  auto valueTable = valueLattice.lookup(drive.op.getValue());
  valueTable.addCondition(condition);
  LLVM_DEBUG({
    llvm::dbgs() << "  - Condition: " << condition << "\n";
    llvm::dbgs() << "  - Value: " << valueTable << "\n";
  });

  // Determine to which edges the drive is sensitive, and ensure that each value
  // is only triggered by a single edge.
  SmallMapVector<Value, Edge, 2> edges;
  for (auto &entry : valueTable.entries) {
    for (auto &orTerm : entry.condition.orTerms) {
      bool edgeSeen = false;
      for (auto &andTerm : orTerm.andTerms) {
        // We only care about values sampled in the past to determine an edge.
        if (!andTerm.hasAnyUses(AndTerm::PastUses))
          continue;

        // We only allow a few triggers.
        if (!triggers.contains(andTerm.value)) {
          LLVM_DEBUG({
            llvm::dbgs() << "- Aborting: past sample of ";
            andTerm.value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
            llvm::dbgs() << " which is not a trigger\n";
          });
          return false;
        }

        // Determine whether we're sampling a pos or neg edge.
        auto edge = Edge::None;
        if (andTerm.uses == AndTerm::PosEdgeUses) {
          edge = Edge::Pos;
        } else if (andTerm.uses == AndTerm::NegEdgeUses) {
          edge = Edge::Neg;
        } else {
          LLVM_DEBUG({
            llvm::dbgs() << "- Aborting: sampling of ";
            andTerm.value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
            llvm::dbgs() << " is neither not a pos/neg edge\n";
          });
          return false;
        }

        // Registers can only react to a single edge. They cannot check for a
        // conjunction of edges.
        if (edgeSeen) {
          LLVM_DEBUG(llvm::dbgs() << "- Aborting: AND of multiple edges\n");
          return false;
        }
        edgeSeen = true;

        // Ensure that we don't sample both a pos and a neg edge.
        auto &existingEdge = edges[andTerm.value];
        if (existingEdge != Edge::None && existingEdge != edge) {
          LLVM_DEBUG({
            llvm::dbgs() << "- Aborting: ";
            andTerm.value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
            llvm::dbgs() << " used both as pos and neg edge trigger\n";
          });
          return false;
        }
        existingEdge = edge;
      }

      if (!edgeSeen) {
        LLVM_DEBUG(llvm::dbgs() << "- Aborting: triggered by non-edge\n");
        return false;
      }
    }
  }
  LLVM_DEBUG({
    for (auto [value, edge] : edges) {
      llvm::dbgs() << "  - Triggered by ";
      value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << (edge == Edge::Pos ? " posedge\n" : " negedge\n");
    }
  });

  // Detect if one of the triggers is a reset. Resets add a particular pattern
  // into the table of drive values, which looks like this:
  // - pos: /rst | rst&... -> const, !rst&... -> ...
  // - neg: \rst | !rst&... -> const, rst&... -> ...
  SmallVector<AndTerm, 2> resetEntries;
  SmallPtrSet<Value, 2> resetTriggers;
  SmallVector<ResetInfo, 2> resetInfos;

  if (edges.size() > 1) {
    for (auto [value, edge] : edges) {
      auto resetEdge =
          edge == Edge::Pos ? AndTerm::posEdge(value) : AndTerm::negEdge(value);
      auto resetActive = AndTerm(
          value, 1 << (edge == Edge::Pos ? AndTerm::Id : AndTerm::NotId));
      auto resetInactive = AndTerm(
          value, 1 << (edge == Edge::Pos ? AndTerm::NotId : AndTerm::Id));

      // Find the single table entry with an edge on the reset.
      unsigned singleResetEntry = -1;
      for (auto [index, entry] : llvm::enumerate(valueTable.entries)) {
        if (!entry.condition.contains(OrTerm(resetEdge)))
          continue;
        // If we have multiple entries with the reset edge as trigger, abort.
        if (singleResetEntry != unsigned(-1)) {
          singleResetEntry = -1;
          break;
        }
        singleResetEntry = index;
      }
      if (singleResetEntry == unsigned(-1))
        continue;

      // The reset value must be a constant. Unfortunately, we don't fold all
      // possible aggregates down to a single constant materializer in the IR.
      // To deal with this, we merely limit reset values to be static: as long
      // as they are derived through side-effect-free operations that only
      // depend on constants, we admit a value as a reset.
      //
      // Technically also non-constants can be reset values. However, since
      // async resets are level sensitive (even though Verilog describes them as
      // edge sensitive), the reset value would have to be part of the wait op's
      // observed values. We don't check for that.
      auto resetValue = valueTable.entries[singleResetEntry].value;
      if (!isValidResetValue(resetValue))
        continue;

      // Ensure that all other entries besides the reset entry contain the
      // inactive reset value as an AND term.
      auto allGated = llvm::all_of(
          llvm::enumerate(valueTable.entries), [&](auto indexAndEntry) {
            // Skip the reset entry.
            if (indexAndEntry.index() == singleResetEntry)
              return true;
            // All other entries must contain the inactive reset.
            return llvm::all_of(
                indexAndEntry.value().condition.orTerms,
                [&](auto &orTerm) { return orTerm.contains(resetInactive); });
          });
      if (!allGated)
        continue;

      // Keep track of the reset.
      resetEntries.push_back(resetEdge);
      resetEntries.push_back(resetActive);
      resetInfos.push_back({value, resetValue, edge == Edge::Pos});
      resetTriggers.insert(value);
    }
  }

  // Remove the resets from the edge triggers.
  for (auto &resetInfo : resetInfos)
    edges.erase(resetInfo.reset);

  // Remove the conditions in the drive value table corresponding to the reset
  // being active. This information has been absorbed into `resetInfos`.
  llvm::erase_if(valueTable.entries, [&](auto &entry) {
    llvm::erase_if(entry.condition.orTerms, [&](auto &orTerm) {
      for (auto &andTerm : orTerm.andTerms)
        if (llvm::is_contained(resetEntries, andTerm))
          return true;
      return false;
    });
    return entry.condition.isFalse();
  });

  // Assume the resets are inactive by removing them from the remaining AND
  // terms in the drive value table.
  for (auto &entry : valueTable.entries) {
    for (auto &orTerm : entry.condition.orTerms) {
      llvm::erase_if(orTerm.andTerms, [&](auto &andTerm) {
        return resetTriggers.contains(andTerm.value);
      });
    }
  }

  // Group the value table entries by the remaining triggers.
  SmallVector<ClockInfo, 2> clockInfos;

  for (auto [clock, edge] : edges) {
    auto clockEdge =
        edge == Edge::Pos ? AndTerm::posEdge(clock) : AndTerm::negEdge(clock);
    ValueTable clockValueTable;

    for (auto &entry : valueTable.entries) {
      auto condition = entry.condition;

      // Remove OR terms in the condition that don't mention this clock as
      // trigger. Also remove the explicit mention of the clock edge from the
      // condition, since this is now implied by the fact that we are grouping
      // the driven values by the clock.
      llvm::erase_if(condition.orTerms, [&](auto &orTerm) {
        bool clockEdgeFound = false;
        llvm::erase_if(orTerm.andTerms, [&](auto &andTerm) {
          if (andTerm == clockEdge) {
            clockEdgeFound = true;
            return true;
          }
          return false;
        });
        return !clockEdgeFound;
      });

      // Check if the condition has become trivially true.
      if (llvm::any_of(condition.orTerms,
                       [](auto &orTerm) { return orTerm.isTrue(); }))
        condition = DNF(true);

      // If the condition is not trivially false, add an entry to the table.
      if (!condition.isFalse())
        clockValueTable.entries.push_back({condition, entry.value});
    }

    // Keep track of the clocks.
    clockInfos.push_back({clock, edge, std::move(clockValueTable)});
  }

  // Handle the degenerate case where the clock changes a register to the same
  // value as the reset. We detect the clock as a reset in this case, leading to
  // zero clocks and two resets. Handle this by promoting one of the resets to a
  // clock.
  if (resetInfos.size() == 2 && clockInfos.empty()) {
    // Guess a sensible clock. Prefer posedge clocks and zero-valued resets.
    // Prefer the first trigger in the wait op's observed value list as the
    // clock.
    unsigned clockIdx = 0;
    if (resetInfos[0].activeHigh && !resetInfos[1].activeHigh)
      clockIdx = 0;
    else if (!resetInfos[0].activeHigh && resetInfos[1].activeHigh)
      clockIdx = 1;
    else if (matchPattern(resetInfos[1].value, m_Zero()))
      clockIdx = 0;
    else if (matchPattern(resetInfos[0].value, m_Zero()))
      clockIdx = 1;
    else if (resetInfos[0].reset == triggers[0])
      clockIdx = 0;
    else if (resetInfos[1].reset == triggers[0])
      clockIdx = 1;

    // Move the clock from `resetInfos` over into the `clockInfos` list.
    auto &info = resetInfos[clockIdx];
    LLVM_DEBUG({
      llvm::dbgs() << "  - Two resets, no clock: promoting ";
      info.reset.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " to clock\n";
    });
    clockInfos.push_back({info.reset, info.activeHigh ? Edge::Pos : Edge::Neg,
                          ValueTable(info.value)});
    resetInfos.erase(resetInfos.begin() + clockIdx);
  }

  // Dump out some debugging information about the detected resets and clocks.
  LLVM_DEBUG({
    for (auto &resetInfo : resetInfos) {
      llvm::dbgs() << "  - Reset ";
      resetInfo.reset.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " (active " << (resetInfo.activeHigh ? "high" : "low")
                   << "): ";
      resetInfo.value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << "\n";
    }
    for (auto &clockInfo : clockInfos) {
      llvm::dbgs() << "  - Clock ";
      clockInfo.clock.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " ("
                   << (clockInfo.edge == Edge::Pos ? "posedge" : "negedge")
                   << "): " << clockInfo.valueTable << "\n";
    }
  });

  // Ensure that the clock and reset is available outside the process.
  for (auto &clockInfo : clockInfos) {
    if (clockInfo.clock.getParentRegion()->isProperAncestor(&process.getBody()))
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "- Aborting: clock ";
      clockInfo.clock.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " not available outside the process\n";
    });
    return false;
  }
  for (auto &resetInfo : resetInfos) {
    if (resetInfo.reset.getParentRegion()->isProperAncestor(&process.getBody()))
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "- Aborting: reset ";
      resetInfo.reset.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " not available outside the process\n";
    });
    return false;
  }

  // The registers we can lower to only support a single clock, and a single
  // optional reset.
  if (clockInfos.size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Aborting: " << clockInfos.size() << " clocks\n");
    return false;
  }
  if (resetInfos.size() > 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Aborting: " << resetInfos.size() << " resets\n");
    return false;
  }
  drive.clock = clockInfos[0];
  drive.reset = resetInfos.empty() ? ResetInfo{} : resetInfos[0];

  return true;
}

/// Check if a value is constant or only derived from constant values through
/// side-effect-free operations. If it is, the value is guaranteed to never
/// change.
bool Deseq::isValidResetValue(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  if (auto it = staticOps.find(result.getOwner()); it != staticOps.end())
    return it->second;

  struct WorklistItem {
    Operation *op;
    OperandRange::iterator it;
  };
  SmallVector<WorklistItem> worklist;
  worklist.push_back({result.getOwner(), result.getOwner()->operand_begin()});

  while (!worklist.empty()) {
    auto item = worklist.pop_back_val();
    auto &isStatic = staticOps[item.op];
    if (item.it == item.op->operand_begin()) {
      if (item.op->hasTrait<OpTrait::ConstantLike>()) {
        isStatic = true;
        continue;
      }
      if (!isMemoryEffectFree(item.op))
        continue;
    }
    if (item.it == item.op->operand_end()) {
      isStatic = true;
      continue;
    } else {
      auto result = dyn_cast<OpResult>(*item.it);
      if (!result)
        continue;
      auto it = staticOps.find(result.getOwner());
      if (it == staticOps.end()) {
        worklist.push_back(item);
        worklist.push_back(
            {result.getOwner(), result.getOwner()->operand_begin()});
      } else if (it->second) {
        ++item.it;
        worklist.push_back(item);
      }
    }
  }

  return staticOps.lookup(result.getOwner());
}

//===----------------------------------------------------------------------===//
// Register Implementation
//===----------------------------------------------------------------------===//

/// Make all drives unconditional and implement the conditional behavior with
/// register ops.
void Deseq::implementRegisters() {
  for (auto &drive : driveInfos)
    implementRegister(drive);
}

/// Implement the conditional behavior of a drive with a `seq.compreg` or
/// `seq.compreg.ce` op, and make the drive unconditional. This function pulls
/// the analyzed clock and reset from the given `DriveInfo` and creates the
/// necessary ops outside the process represent the behavior as a register. It
/// also calls `specializeValue` and `specializeProcess` to convert the
/// sequential `llhd.process` into a purely combinational `scf.execute_region`
/// that is simplified by assuming that the clock edge occurs.
void Deseq::implementRegister(DriveInfo &drive) {
  OpBuilder builder(drive.op);
  auto loc = drive.op.getLoc();

  // Materialize the clock as a `!seq.clock` value. Insert an inverter for
  // negedge clocks.
  Value clock = builder.create<seq::ToClockOp>(loc, drive.clock.clock);
  if (drive.clock.edge == Edge::Neg)
    clock = builder.create<seq::ClockInverterOp>(loc, clock);

  // Handle the optional reset.
  Value reset;
  Value resetValue;

  if (drive.reset) {
    reset = drive.reset.reset;
    resetValue = drive.reset.value;

    // Materialize the reset as an `i1` value. Insert an inverter for negedge
    // resets.
    if (!drive.reset.activeHigh) {
      auto one = builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
      reset = builder.create<comb::XorOp>(loc, reset, one);
    }

    // Specialize the process for the reset trigger. If the reset value is
    // trivially available outside the process, use it directly. If it is a
    // constant, move the constant outside the process.
    if (!resetValue.getParentRegion()->isProperAncestor(&process.getBody())) {
      if (auto *defOp = resetValue.getDefiningOp();
          defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
        defOp->moveBefore(process);
      } else {
        resetValue = specializeValue(
            drive.op.getValue(),
            FixedValues{{drive.clock.clock, drive.clock.edge == Edge::Neg,
                         drive.clock.edge == Edge::Neg},
                        {drive.reset.reset, !drive.reset.activeHigh,
                         drive.reset.activeHigh}});
      }
    }
  }

  // Determine the enable condition. If we have determined that the register
  // is trivially enabled, don't add an enable. If the enable condition is a
  // simple boolean value available outside the process, use it directly.
  Value enable = drive.op.getEnable();
  if (drive.clock.valueTable.entries.size() == 1) {
    if (drive.clock.valueTable.entries[0].condition.isTrue())
      enable = {};
    else if (auto singleTerm =
                 drive.clock.valueTable.entries[0].condition.getSingleTerm();
             singleTerm && singleTerm->second == AndTerm::Id &&
             singleTerm->first.getParentRegion()->isProperAncestor(
                 &process.getBody())) {
      enable = singleTerm->first;
    }
  }

  // Determine the value. If the value is trivially available outside the
  // process, use it directly. If it is a constant, move the constant outside
  // the process.
  Value value = drive.op.getValue();
  if (drive.clock.valueTable.entries.size() == 1) {
    auto tryValue = drive.clock.valueTable.entries[0].value;
    if (tryValue.getParentRegion()->isProperAncestor(&process.getBody())) {
      value = tryValue;
    } else if (auto *defOp = tryValue.getDefiningOp();
               defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
      defOp->moveBefore(process);
      value = tryValue;
    }
  }

  // Specialize the process for the clock trigger, which will produce the
  // enable and the value for regular clock edges.
  FixedValues fixedValues;
  fixedValues.push_back({drive.clock.clock, drive.clock.edge == Edge::Neg,
                         drive.clock.edge == Edge::Pos});
  if (drive.reset)
    fixedValues.push_back(
        {drive.reset.reset, !drive.reset.activeHigh, !drive.reset.activeHigh});

  value = specializeValue(value, fixedValues);
  if (enable)
    enable = specializeValue(enable, fixedValues);

  // Create the register op.
  Value reg;
  if (enable)
    reg = builder.create<seq::CompRegClockEnabledOp>(
        loc, value, clock, enable, StringAttr{}, reset, resetValue, Value{},
        hw::InnerSymAttr{});
  else
    reg =
        builder.create<seq::CompRegOp>(loc, value, clock, StringAttr{}, reset,
                                       resetValue, Value{}, hw::InnerSymAttr{});

  // Make the original `llhd.drv` drive the register value unconditionally.
  drive.op.getValueMutable().assign(reg);
  drive.op.getEnableMutable().clear();

  // If the original `llhd.drv` had a delta delay, turn it into an immediate
  // drive since the delay behavior is now capture by the register op.
  TimeAttr attr;
  if (matchPattern(drive.op.getTime(), m_Constant(&attr)) &&
      attr.getTime() == 0 && attr.getDelta() == 1 && attr.getEpsilon() == 0) {
    if (!epsilonDelay)
      epsilonDelay =
          builder.create<ConstantTimeOp>(process.getLoc(), 0, "ns", 0, 1);
    drive.op.getTimeMutable().assign(epsilonDelay);
  }
}

//===----------------------------------------------------------------------===//
// Process Specialization
//===----------------------------------------------------------------------===//

/// Specialize a value by assuming the values listed in `fixedValues` are at a
/// constant value in the past and the present. The function is guaranteed to
/// replace results of the process with results of a new combinational
/// `scf.execute_region` op. All other behavior is purely an optimization; the
/// function may not make use of the assignments in `fixedValues` at all.
Value Deseq::specializeValue(Value value, FixedValues fixedValues) {
  auto result = dyn_cast<OpResult>(value);
  if (!result || result.getOwner() != process)
    return value;
  return specializeProcess(fixedValues)[result.getResultNumber()];
}

/// Specialize the current process by assuming the values listed in
/// `fixedValues` are at a constant value in the past and the present. This
/// function creates a new `scf.execute_region` op with a simplified version
/// of the process where all uses of the values listed in `fixedValues` are
/// replaced with their constant counterpart. Since the clock-dependent
/// behavior of the process has been absorbed into a register, the process can
/// be replaced with a combinational representation that computes the drive
/// value and drive condition under the assumption that the clock edge occurs.
ValueRange Deseq::specializeProcess(FixedValues fixedValues) {
  if (auto it = specializedProcesses.find(fixedValues);
      it != specializedProcesses.end())
    return it->second;

  LLVM_DEBUG({
    llvm::dbgs() << "- Specializing process for:\n";
    for (auto fixedValue : fixedValues) {
      llvm::dbgs() << "  - ";
      fixedValue.value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << ": " << fixedValue.past << " -> " << fixedValue.present
                   << "\n";
    }
  });

  // Create an `scf.execute_region` with this process specialized to compute
  // the result for the given fixed values. The triggers will be absorbed into
  // the register operation that consumes the result of this specialized
  // process, such that we can make the process purely combinational.
  OpBuilder builder(process);
  auto executeOp = builder.create<scf::ExecuteRegionOp>(
      process.getLoc(), process.getResultTypes());

  IRMapping mapping;
  SmallVector<std::pair<Block *, Block *>> worklist;

  auto scheduleBlock = [&](Block *block) {
    if (auto *newBlock = mapping.lookupOrNull(block))
      return newBlock;
    auto *newBlock = &executeOp.getRegion().emplaceBlock();
    for (auto arg : block->getArguments()) {
      auto newArg = newBlock->addArgument(arg.getType(), arg.getLoc());
      mapping.map(arg, newArg);
    }
    mapping.map(block, newBlock);
    worklist.push_back({block, newBlock});
    return newBlock;
  };

  // Initialize the mapping with constants for the fixed values.
  auto &entryBlock = executeOp.getRegion().emplaceBlock();
  builder.setInsertionPointToStart(&entryBlock);
  auto i1 = builder.getI1Type();
  auto trueValue = builder.create<hw::ConstantOp>(process.getLoc(), i1, 1);
  auto falseValue = builder.create<hw::ConstantOp>(process.getLoc(), i1, 0);

  SmallDenseMap<Value, std::pair<Value, Value>, 2> materializedFixedValues;
  for (auto fixedValue : fixedValues) {
    auto present = fixedValue.present ? trueValue : falseValue;
    auto past = fixedValue.past ? trueValue : falseValue;
    materializedFixedValues.insert({fixedValue.value, {past, present}});
    mapping.map(fixedValue.value, present);
  }

  auto evaluateTerm = [&](Value value, bool past) -> std::optional<bool> {
    for (auto fixedValue : fixedValues)
      if (fixedValue.value == value)
        return past ? fixedValue.past : fixedValue.present;
    return {};
  };

  // Clone operations over.
  auto cloneBlocks = [&](bool stopAtWait) {
    SmallVector<Value> foldedResults;
    while (!worklist.empty()) {
      auto [oldBlock, newBlock] = worklist.pop_back_val();
      builder.setInsertionPointToEnd(newBlock);
      for (auto &oldOp : *oldBlock) {
        // Convert `llhd.wait` into `scf.yield`.
        if (auto waitOp = dyn_cast<WaitOp>(oldOp)) {
          if (stopAtWait)
            continue;
          SmallVector<Value> operands;
          for (auto operand : waitOp.getYieldOperands())
            operands.push_back(mapping.lookupOrDefault(operand));
          builder.create<scf::YieldOp>(waitOp.getLoc(), operands);
          continue;
        }

        // Convert `cf.cond_br` ops into `cf.br` if the condition is constant.
        if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(oldOp)) {
          SmallVector<Value> operands;
          auto condition = mapping.lookupOrDefault(condBranchOp.getCondition());
          if (matchPattern(condition, m_NonZero())) {
            for (auto operand : condBranchOp.getTrueDestOperands())
              operands.push_back(mapping.lookupOrDefault(operand));
            builder.create<cf::BranchOp>(
                condBranchOp.getLoc(),
                scheduleBlock(condBranchOp.getTrueDest()), operands);
            continue;
          }
          if (matchPattern(condition, m_Zero())) {
            for (auto operand : condBranchOp.getFalseOperands())
              operands.push_back(mapping.lookupOrDefault(operand));
            builder.create<cf::BranchOp>(
                condBranchOp.getLoc(),
                scheduleBlock(condBranchOp.getFalseDest()), operands);
            continue;
          }
        }

        // If our initial data flow analysis has produced a concrete DNF for
        // an `i1`-valued op, see if the DNF evaluates to a constant true or
        // false with the given fixed values.
        if (oldOp.getNumResults() == 1 &&
            oldOp.getResult(0).getType().isSignlessInteger(1)) {
          if (auto dnf = booleanLattice.lookup(oldOp.getResult(0))) {
            if (auto result = dnf.evaluate(evaluateTerm)) {
              mapping.map(oldOp.getResult(0), *result ? trueValue : falseValue);
              continue;
            }
          }
        }

        // Otherwise clone the operation.
        for (auto &blockOperand : oldOp.getBlockOperands())
          scheduleBlock(blockOperand.get());
        auto *clonedOp = builder.clone(oldOp, mapping);

        // And immediately try to fold the cloned operation since the fixed
        // values introduce a lot of constants into the IR.
        if (succeeded(builder.tryFold(clonedOp, foldedResults)) &&
            !foldedResults.empty()) {
          for (auto [oldResult, foldedResult] :
               llvm::zip(oldOp.getResults(), foldedResults))
            mapping.map(oldResult, foldedResult);
          clonedOp->erase();
        }
        foldedResults.clear();
      }
    }
  };

  // Start at the entry block of the original process and clone all ops until
  // we hit the wait.
  worklist.push_back({&process.getBody().front(), &entryBlock});
  cloneBlocks(true);
  builder.setInsertionPointToEnd(mapping.lookup(wait->getBlock()));

  // Remove all blocks from the IR mapping. Some blocks may be reachable from
  // the entry block and the wait op, in which case we want to create
  // duplicates of those blocks.
  for (auto &block : process.getBody())
    mapping.erase(&block);

  // If the wait op is not the only predecessor of its destination block,
  // create a branch op to the block. Otherwise inline the destination block
  // into the entry block, which allows the specialization to fold more
  // constants.
  if (wait.getDest()->hasOneUse()) {
    // Map the block arguments of the block after the wait op to the constant
    // fixed values.
    for (auto [arg, pastValue] :
         llvm::zip(wait.getDest()->getArguments(), pastValues))
      mapping.map(arg, materializedFixedValues.lookup(pastValue).first);

    // Schedule the block after the wait for cloning into the entry block.
    mapping.map(wait.getDest(), builder.getBlock());
    worklist.push_back({wait.getDest(), builder.getBlock()});
  } else {
    // Schedule the block after the wait for cloning.
    auto *dest = scheduleBlock(wait.getDest());

    // From the entry block, branch to the block after the wait with the
    // appropriate past values as block arguments.
    SmallVector<Value> destOperands;
    assert(pastValues.size() == wait.getDestOperands().size());
    for (auto pastValue : pastValues)
      destOperands.push_back(materializedFixedValues.lookup(pastValue).first);
    builder.create<cf::BranchOp>(wait.getLoc(), dest, destOperands);
  }

  // Clone everything after the wait operation.
  cloneBlocks(false);

  // Don't leave unused constants behind.
  if (isOpTriviallyDead(trueValue))
    trueValue.erase();
  if (isOpTriviallyDead(falseValue))
    falseValue.erase();

  specializedProcesses.insert({fixedValues, executeOp.getResults()});
  return executeOp.getResults();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct DeseqPass : public llhd::impl::DeseqPassBase<DeseqPass> {
  void runOnOperation() override;
};
} // namespace

void DeseqPass::runOnOperation() {
  SmallVector<ProcessOp> processes(getOperation().getOps<ProcessOp>());
  for (auto process : processes)
    Deseq(process).deseq();
}
