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
  /// Create a table with a single value produced under a given condition.
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
  /// Create a reduced table that only contains values which overlap with a
  /// given condition.
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
  /// The value the register is set to when the clock is triggered.
  Value value;
  /// Whether the clock is sensitive to a rising or falling edge.
  bool risingEdge;
  /// The optional value acting as an enable.
  Value enable;

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
  Value tracePastValue(Value pastValue);

  bool matchDrives();
  bool matchDrive(DriveInfo &drive);
  bool matchDriveClock(
      DriveInfo &drive,
      ArrayRef<std::pair<BetterDNFTerm, BetterValueEntry>> valueTable);
  bool matchDriveClockAndReset(
      DriveInfo &drive,
      ArrayRef<std::pair<BetterDNFTerm, BetterValueEntry>> valueTable);
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

  TruthTable computeBoolean(Value value);
  BetterValueTable computeValue(Value value);
  TruthTable computeBoolean(OpResult value);
  BetterValueTable computeValue(OpResult value);
  TruthTable computeBoolean(BlockArgument value);
  BetterValueTable computeValue(BlockArgument arg);
  TruthTable computeBlockCondition(Block *block);
  TruthTable computeSuccessorCondition(BlockOperand &operand);
  TruthTable computeSuccessorBoolean(BlockOperand &operand, unsigned argIdx);
  BetterValueTable computeSuccessorValue(BlockOperand &operand,
                                         unsigned argIdx);

  TruthTable getPoisonBoolean() const { return TruthTable::getPoison(); }

  TruthTable getUnknownBoolean() const {
    return TruthTable::getTerm(observedValues.size() * 2 + 1, 0);
  }

  TruthTable getConstBoolean(bool value) const {
    return TruthTable::getConst(observedValues.size() * 2 + 1, value);
  }

  TruthTable getPastTrigger(unsigned triggerIndex) const {
    return TruthTable::getTerm(observedValues.size() * 2 + 1,
                               triggerIndex * 2 + 1);
  }

  TruthTable getPresentTrigger(unsigned triggerIndex) const {
    return TruthTable::getTerm(observedValues.size() * 2 + 1,
                               triggerIndex * 2 + 2);
  }

  BetterValueTable getUnknownValue() const {
    return BetterValueTable(getConstBoolean(true),
                            BetterValueEntry::getUnknown());
  }

  BetterValueTable getPoisonValue() const {
    return BetterValueTable(getConstBoolean(true),
                            BetterValueEntry::getPoison());
  }

  BetterValueTable getKnownValue(Value value) const {
    return BetterValueTable(getConstBoolean(true), value);
  }

  /// The process we are desequentializing.
  ProcessOp process;
  /// The single wait op of the process.
  WaitOp wait;
  /// The boolean values observed by the wait.
  SmallSetVector<Value, 2> observedValues;
  /// The values carried from the past into the present as destination operands
  /// of the wait op. These values are guaranteed to also be contained in
  /// `observedValues`.
  SmallVector<Value, 2> pastValues;
  /// The conditional drive operations fed by this process.
  SmallVector<DriveInfo> driveInfos;
  /// The triggers that cause the process to update its results.
  SmallSetVector<Value, 2> triggers; // TODO: merge with observedValues
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
  DenseMap<Value, TruthTable> booleanLattice2;
  /// The value table computed for a value in the IR. This essentially lists
  /// what values an SSA value assumes under certain conditions.
  DenseMap<Value, ValueTable> valueLattice;
  DenseMap<Value, BetterValueTable> valueLattice2;
  /// The condition under which control flow reaches a block. The block
  /// immediately following the wait op has this set to true; any further
  /// conditional branches will refine the condition of successor blocks.
  DenseMap<Block *, DNF> blockConditionLattice;
  DenseMap<Block *, TruthTable> blockConditionLattice2;
  /// The conditions and values transferred from a block to its successors.
  DenseMap<Block *, SuccessorValues> successorLattice;
  DenseMap<BlockOperand *, DNF> successorConditionLattice;
  DenseMap<BlockOperand *, TruthTable> successorConditionLattice2;
  DenseMap<std::tuple<BlockOperand *, unsigned>, DNF> successorBooleanLattice;
  DenseMap<std::tuple<BlockOperand *, unsigned>, TruthTable>
      successorBooleanLattice2;
  DenseMap<std::tuple<BlockOperand *, unsigned>, BetterValueTable>
      successorValueLattice2;
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
    llvm::dbgs() << "- " << observedValues.size() << " potential triggers:\n";
    for (auto [index, observedValue] : llvm::enumerate(observedValues)) {
      llvm::dbgs() << "  - ";
      observedValue.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << ": past " << getPastTrigger(index);
      llvm::dbgs() << ", present " << getPresentTrigger(index);
      llvm::dbgs() << "\n";
    }
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
  // We can only desequentialize processes with no side-effecting ops besides
  // the `WaitOp` or `HaltOp` terminators.
  for (auto &block : process.getBody()) {
    for (auto &op : block) {
      if (isa<WaitOp, HaltOp>(op))
        continue;
      if (!isMemoryEffectFree(&op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "Skipping " << process.getLoc()
                       << ": contains side-effecting op ";
          op.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
          llvm::dbgs() << "\n";
        });
        return false;
      }
    }
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

  // Ensure the observed values are all booleans.
  for (auto value : wait.getObserved()) {
    if (!value.getType().isSignlessInteger(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                              << ": observes non-i1 value\n");
      return false;
    }
    observedValues.insert(value);
  }

  // We only support 1 or 2 observed values, since we map to registers with a
  // clock and an optional async reset.
  if (observedValues.empty() || observedValues.size() > 2) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc() << ": observes "
                            << observedValues.size() << " values\n");
    return false;
  }

  // Seed the drive value analysis with the observed values.
  for (auto [index, observedValue] : llvm::enumerate(observedValues))
    booleanLattice2.insert({observedValue, getPresentTrigger(index)});

  // Ensure the wait op destination operands, i.e. the values passed from the
  // past into the present, are the observed values.
  for (auto [operand, blockArg] :
       llvm::zip(wait.getDestOperands(), wait.getDest()->getArguments())) {
    if (!operand.getType().isSignlessInteger(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                              << ": uses non-i1 past value\n");
      return false;
    }
    auto observedValue = tracePastValue(operand);
    if (!observedValue)
      return false;
    pastValues.push_back(observedValue);
    unsigned index = std::distance(observedValues.begin(),
                                   llvm::find(observedValues, observedValue));
    booleanLattice2.insert({blockArg, getPastTrigger(index)});
  }

  return true;
}

/// Trace a value passed from the past into the present as a destination operand
/// of the wait op back to a single observed value. Returns a null value if the
/// value does not trace back to a single, unique observed value.
Value Deseq::tracePastValue(Value pastValue) {
  // Use a worklist to look through branches and a few common IR patterns to
  // find the concrete value used as a destination operand.
  SmallVector<Value> worklist;
  SmallPtrSet<Value, 8> seen;
  worklist.push_back(pastValue);
  seen.insert(pastValue);

  SmallPtrSet<Block *, 2> predSeen;
  SmallSetVector<BlockOperand *, 4> predWorklist;
  SmallPtrSet<Value, 2> distinctValues;
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    auto arg = dyn_cast<BlockArgument>(value);

    // If this is one of the observed values, we're done. Otherwise trace
    // block arguments backwards to their predecessors.
    if (observedValues.contains(value) || !arg) {
      distinctValues.insert(value);
      continue;
    }

    // Collect the predecessor block operands to process.
    predSeen.clear();
    predWorklist.clear();
    for (auto *predecessor : arg.getOwner()->getPredecessors())
      if (predSeen.insert(predecessor).second)
        for (auto &operand : predecessor->getTerminator()->getBlockOperands())
          if (operand.get() == arg.getOwner())
            predWorklist.insert(&operand);

    // Handle the predecessors. This essentially is a loop over all block
    // arguments in terminator ops that branch to arg's block.
    unsigned argIdx = arg.getArgNumber();
    for (auto *blockOperand : predWorklist) {
      auto *op = blockOperand->getOwner();
      if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
        // Handle unconditional branches.
        auto operand = branchOp.getDestOperands()[argIdx];
        if (seen.insert(operand).second)
          worklist.push_back(operand);
      } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
        // Handle conditional branches.
        unsigned destIdx = blockOperand->getOperandNumber();
        auto operand = destIdx == 0
                           ? condBranchOp.getTrueDestOperands()[argIdx]
                           : condBranchOp.getFalseDestOperands()[argIdx];

        // Undo the `cond_br a, bb(a), bb(a)` to `cond_br a, bb(1), bb(0)`
        // canonicalization.
        if ((matchPattern(operand, m_One()) && destIdx == 0) ||
            (matchPattern(operand, m_Zero()) && destIdx == 1))
          operand = condBranchOp.getCondition();

        if (seen.insert(operand).second)
          worklist.push_back(operand);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                                << ": unsupported terminator " << op->getName()
                                << " while tracing past value\n");
        return Value{};
      }
    }
  }

  // Ensure that we have one distinct value being passed from the past into
  // the present, and that the value is observed.
  if (distinctValues.size() != 1) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Skipping " << process.getLoc()
        << ": multiple past values passed for the same block argument\n");
    return Value{};
  }
  auto distinctValue = *distinctValues.begin();
  if (!observedValues.contains(distinctValue)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                            << ": unobserved past value\n");
    return Value{};
  }
  return distinctValue;
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
            defOp && !observedValues.contains(value)) {
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
// Data Flow Analysis (Updated)
//===----------------------------------------------------------------------===//

TruthTable Deseq::computeBoolean(Value value) {
  assert(value.getType().isSignlessInteger(1));

  // If this value is a result of the process we're analyzing, jump to the
  // corresponding yield operand of the wait op.
  if (value.getDefiningOp() == process)
    return computeBoolean(
        wait.getYieldOperands()[cast<OpResult>(value).getResultNumber()]);

  // Check if we have already computed this value. Otherwise insert an unknown
  // value to break recursions. This will be overwritten by a concrete value
  // later.
  if (auto it = booleanLattice2.find(value); it != booleanLattice2.end())
    return it->second;
  booleanLattice2[value] = getUnknownBoolean();

  // Actually compute the value.
  TruthTable result =
      TypeSwitch<Value, TruthTable>(value).Case<OpResult, BlockArgument>(
          [&](auto value) { return computeBoolean(value); });

  // Memoize the result.
  LLVM_DEBUG({
    llvm::dbgs() << "- Boolean ";
    value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
    llvm::dbgs() << ": " << result << "\n";
  });
  booleanLattice2[value] = result;
  return result;
}

BetterValueTable Deseq::computeValue(Value value) {
  // If this value is a result of the process we're analyzing, jump to the
  // corresponding yield operand of the wait op.
  if (value.getDefiningOp() == process)
    return computeValue(
        wait.getYieldOperands()[cast<OpResult>(value).getResultNumber()]);

  // Check if we have already computed this value. Otherwise insert an unknown
  // value to break recursions. This will be overwritten by a concrete value
  // later.
  if (auto it = valueLattice2.find(value); it != valueLattice2.end())
    return it->second;
  valueLattice2[value] = getUnknownValue();

  // Actually compute the value.
  BetterValueTable result =
      TypeSwitch<Value, BetterValueTable>(value).Case<OpResult, BlockArgument>(
          [&](auto value) { return computeValue(value); });

  // Memoize the result.
  LLVM_DEBUG({
    llvm::dbgs() << "- Value ";
    value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
    llvm::dbgs() << ": " << result << "\n";
  });
  valueLattice2[value] = result;
  return result;
}

TruthTable Deseq::computeBoolean(OpResult value) {
  assert(value.getType().isSignlessInteger(1));
  auto *op = value.getOwner();

  // Handle constants.
  if (auto constOp = dyn_cast<hw::ConstantOp>(op))
    return getConstBoolean(constOp.getValue().isOne());

  // Handle `comb.or`.
  if (auto orOp = dyn_cast<comb::OrOp>(op)) {
    auto result = getConstBoolean(false);
    for (auto operand : orOp.getInputs()) {
      result |= computeBoolean(operand);
      if (result.isTrue())
        break;
    }
    return result;
  }

  // Handle `comb.and`.
  if (auto andOp = dyn_cast<comb::AndOp>(op)) {
    auto result = getConstBoolean(true);
    for (auto operand : andOp.getInputs()) {
      result &= computeBoolean(operand);
      if (result.isFalse())
        break;
    }
    return result;
  }

  // Handle `comb.xor`.
  if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
    auto result = getConstBoolean(false);
    for (auto operand : xorOp.getInputs())
      result ^= computeBoolean(operand);
    return result;
  }

  // Otherwise check if the operation depends on any of the triggers. If it
  // does, create a poison value since we don't really know how the trigger
  // affects this boolean. If it doesn't, create an unknown value.
  if (llvm::any_of(op->getOperands(), [&](auto operand) {
        // TODO: This should probably also check non-i1 values to see if they
        // depend on the triggers. Maybe once we merge boolean and value tables?
        if (!operand.getType().isSignlessInteger(1))
          return false;
        auto result = computeBoolean(operand);
        return result.isPoison() || result != getUnknownBoolean();
      }))
    return getPoisonBoolean();
  return getUnknownBoolean();
}

BetterValueTable Deseq::computeValue(OpResult value) {
  // // Handle constants.
  // Attribute attr;
  // if (matchPattern(value, m_Constant(&attr)))
  //   return getConstValue(attr);

  // TODO: Support comb.mux and arith.select?
  // TODO: Reject values that depend on the triggers.
  return getKnownValue(value);
}

TruthTable Deseq::computeBoolean(BlockArgument arg) {
  auto *block = arg.getOwner();

  // If this isn't a block in the process, simply return an unknown value.
  if (block->getParentOp() != process)
    return getUnknownBoolean();

  // Otherwise iterate over all predecessors and compute the boolean values
  // being passed to this block argument by each.
  auto result = getConstBoolean(false);
  SmallPtrSet<Block *, 4> seen;
  for (auto *predecessor : block->getPredecessors()) {
    if (!seen.insert(predecessor).second)
      continue;
    for (auto &operand : predecessor->getTerminator()->getBlockOperands()) {
      if (operand.get() != block)
        continue;
      auto value = computeSuccessorBoolean(operand, arg.getArgNumber());
      if (value.isFalse())
        continue;
      auto condition = computeSuccessorCondition(operand);
      result |= value & condition;
      if (result.isTrue())
        break;
    }
    if (result.isTrue())
      break;
  }
  return result;
}

BetterValueTable Deseq::computeValue(BlockArgument arg) {
  auto *block = arg.getOwner();

  // If this isn't a block in the process, simply return the value itself.
  if (block->getParentOp() != process)
    return getKnownValue(arg);

  // Otherwise iterate over all predecessors and compute the boolean values
  // being passed to this block argument by each.
  auto result = BetterValueTable();
  SmallPtrSet<Block *, 4> seen;
  for (auto *predecessor : block->getPredecessors()) {
    if (!seen.insert(predecessor).second)
      continue;
    for (auto &operand : predecessor->getTerminator()->getBlockOperands()) {
      if (operand.get() != block)
        continue;
      auto condition = computeSuccessorCondition(operand);
      if (condition.isFalse())
        continue;
      auto value = computeSuccessorValue(operand, arg.getArgNumber());
      value.addCondition(condition);
      result.merge(value);
    }
  }
  return result;
}

TruthTable Deseq::computeBlockCondition(Block *block) {
  // Return a memoized result if one exists. Otherwise insert a default result
  // as recursion breaker.
  if (auto it = blockConditionLattice2.find(block);
      it != blockConditionLattice2.end())
    return it->second;
  blockConditionLattice2[block] = getUnknownBoolean();

  // Actually compute the block condition by combining all incoming control flow
  // conditions.
  auto result = getConstBoolean(false);
  SmallPtrSet<Block *, 4> seen;
  for (auto *predecessor : block->getPredecessors()) {
    if (!seen.insert(predecessor).second)
      continue;
    for (auto &operand : predecessor->getTerminator()->getBlockOperands()) {
      if (operand.get() != block)
        continue;
      result |= computeSuccessorCondition(operand);
      if (result.isTrue())
        break;
    }
    if (result.isTrue())
      break;
  }

  // Memoize the result.
  LLVM_DEBUG({
    llvm::dbgs() << "- Block condition ";
    block->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ": " << result << "\n";
  });
  blockConditionLattice2[block] = result;
  return result;
}

TruthTable Deseq::computeSuccessorCondition(BlockOperand &blockOperand) {
  // The wait operation of the process is the origin point of the analysis. We
  // want to know under which conditions drives happen once the wait resumes.
  // Therefore the branch from the wait to its destination block is expected to
  // happen.
  auto *op = blockOperand.getOwner();
  if (op == wait)
    return getConstBoolean(true);

  // Return a memoized result if one exists. Otherwise insert a default result
  // as recursion breaker.
  if (auto it = successorConditionLattice2.find(&blockOperand);
      it != successorConditionLattice2.end())
    return it->second;
  successorConditionLattice2[&blockOperand] = getUnknownBoolean();

  // Actually compute the condition under which control flows along the given
  // block operand.
  auto destIdx = blockOperand.getOperandNumber();
  auto blockCondition = computeBlockCondition(op->getBlock());
  auto result = getUnknownBoolean();
  if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
    result = blockCondition;
  } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
    auto branchCondition = computeBoolean(condBranchOp.getCondition());
    if (destIdx == 0)
      result = blockCondition & branchCondition;
    else
      result = blockCondition & ~branchCondition;
  } else {
    op->emitOpError("not supported in desequentialization");
    result = getPoisonBoolean();
  }

  // Memoize the result.
  LLVM_DEBUG({
    llvm::dbgs() << "- Successor condition ";
    op->getBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#succ" << destIdx << " -> ";
    blockOperand.get()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " = " << result << "\n";
  });
  successorConditionLattice2[&blockOperand] = result;
  return result;
}

TruthTable Deseq::computeSuccessorBoolean(BlockOperand &blockOperand,
                                          unsigned argIdx) {
  // Return a memoized result if one exists. Otherwise insert a default result
  // as recursion breaker.
  if (auto it = successorBooleanLattice2.find({&blockOperand, argIdx});
      it != successorBooleanLattice2.end())
    return it->second;
  successorBooleanLattice2[{&blockOperand, argIdx}] = getUnknownBoolean();

  // Actually compute the boolean destination operand for the given destination
  // block.
  auto *op = blockOperand.getOwner();
  auto destIdx = blockOperand.getOperandNumber();
  auto result = getUnknownBoolean();
  if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
    result = computeBoolean(branchOp.getDestOperands()[argIdx]);
  } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
    if (destIdx == 0)
      result = computeBoolean(condBranchOp.getTrueDestOperands()[argIdx]);
    else
      result = computeBoolean(condBranchOp.getFalseDestOperands()[argIdx]);
  } else {
    op->emitOpError("not supported in desequentialization");
    result = getPoisonBoolean();
  }

  // Memoize the result.
  LLVM_DEBUG({
    llvm::dbgs() << "- Successor boolean ";
    op->getBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#succ" << destIdx << " -> ";
    blockOperand.get()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#arg" << argIdx << " = " << result << "\n";
  });
  successorBooleanLattice2[{&blockOperand, argIdx}] = result;
  return result;
}

BetterValueTable Deseq::computeSuccessorValue(BlockOperand &blockOperand,
                                              unsigned argIdx) {
  // Return a memoized result if one exists. Otherwise insert a default result
  // as recursion breaker.
  if (auto it = successorValueLattice2.find({&blockOperand, argIdx});
      it != successorValueLattice2.end())
    return it->second;
  successorValueLattice2[{&blockOperand, argIdx}] = getUnknownValue();

  // Actually compute the boolean destination operand for the given destination
  // block.
  auto *op = blockOperand.getOwner();
  auto destIdx = blockOperand.getOperandNumber();
  auto result = getUnknownValue();
  if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
    result = computeValue(branchOp.getDestOperands()[argIdx]);
  } else if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
    if (destIdx == 0)
      result = computeValue(condBranchOp.getTrueDestOperands()[argIdx]);
    else
      result = computeValue(condBranchOp.getFalseDestOperands()[argIdx]);
  } else {
    op->emitOpError("not supported in desequentialization");
    result = getPoisonValue();
  }

  // Memoize the result.
  LLVM_DEBUG({
    llvm::dbgs() << "- Successor value ";
    op->getBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#succ" << destIdx << " -> ";
    blockOperand.get()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#arg" << argIdx << " = " << result << "\n";
  });
  successorValueLattice2[{&blockOperand, argIdx}] = result;
  return result;
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
/// determined by the data flow analysis is implementable by a register op. The
/// results are stored in the clock and reset info of the given `DriveInfo`.
/// Returns false if the drive cannot be implemented as a register.
bool Deseq::matchDrive(DriveInfo &drive) {
  LLVM_DEBUG(llvm::dbgs() << "- Analyzing " << drive.op << "\n");

  // Determine under which condition the drive is enabled.
  auto condition = computeBoolean(drive.op.getEnable());
  if (condition.isPoison()) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Aborting: poison condition on " << drive.op << "\n");
    return false;
  }

  // Determine which value is driven under which conditions.
  auto initialValueTable = computeValue(drive.op.getValue());
  initialValueTable.addCondition(condition);
  LLVM_DEBUG({
    llvm::dbgs() << "  - Condition: " << condition << "\n";
    llvm::dbgs() << "  - Value: " << initialValueTable << "\n";
  });

  // Convert the value table from having DNF conditions to having DNFTerm
  // conditions. This effectively spreads OR operations in the conditions across
  // multiple table entries.
  SmallVector<std::pair<BetterDNFTerm, BetterValueEntry>> valueTable;
  for (auto &[condition, value] : initialValueTable.entries) {
    auto dnf = condition.canonicalize();
    if (dnf.isPoison() || value.isPoison()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Aborting: poison in " << initialValueTable << "\n");
      return false;
    }
    for (auto &orTerm : dnf.orTerms)
      valueTable.push_back({orTerm, value});
  }

  // At this point we should have at most three entries in the value table,
  // corresponding to the reset, clock, and clock under reset. Everything else
  // we have no chance of representing as a register op.
  if (valueTable.size() > 3) {
    LLVM_DEBUG(llvm::dbgs() << "- Aborting: value table has "
                            << valueTable.size() << " distinct conditions\n");
    return false;
  }

  // If we have two triggers, one of them must be the reset.
  if (observedValues.size() == 2)
    return matchDriveClockAndReset(drive, valueTable);

  // Otherwise we only have a single trigger, which is the clock.
  assert(observedValues.size() == 1);
  return matchDriveClock(drive, valueTable);
}

/// Assuming there is one trigger, detect the clock scheme represented by a
/// value table and store the results in `drive.clock`.
bool Deseq::matchDriveClock(
    DriveInfo &drive,
    ArrayRef<std::pair<BetterDNFTerm, BetterValueEntry>> valueTable) {
  // We need exactly one entry in the value table to represent a register
  // without reset.
  if (valueTable.size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "- Aborting: single trigger value table has "
                            << valueTable.size() << " entries\n");
    return false;
  }

  // Try the posedge and negedge variants of clocking.
  for (unsigned variant = 0; variant < (1 << 1); ++variant) {
    bool negClock = (variant >> 0) & 1;

    // Assemble the conditions in the value table corresponding to a clock edge
    // with and without an additional enable condition. The enable condition is
    // represented as an additional unknown AND term. The bit patterns here
    // follow from how we assign indices to past and present triggers, and how
    // the DNF's even bits represent positive terms and odd bits represent
    // inverted terms.
    uint32_t clockEdge = (negClock ? 0b1001 : 0b0110) << 2;
    auto clockWithoutEnable = BetterDNFTerm{clockEdge};
    auto clockWithEnable = BetterDNFTerm{clockEdge | 0b01};

    // Check if the single value table entry matches this clock.
    if (valueTable[0].first == clockWithEnable)
      drive.clock.enable = drive.op.getEnable();
    else if (valueTable[0].first != clockWithoutEnable)
      continue;

    // Populate the clock info and return.
    drive.clock.clock = observedValues[0];
    drive.clock.risingEdge = !negClock;
    drive.clock.value = drive.op.getValue();
    if (!valueTable[0].second.isUnknown())
      drive.clock.value = valueTable[0].second.value;

    LLVM_DEBUG({
      llvm::dbgs() << "  - Matched " << (negClock ? "neg" : "pos")
                   << "edge clock ";
      drive.clock.clock.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " -> " << valueTable[0].second;
      if (drive.clock.enable)
        llvm::dbgs() << " (with enable)";
      llvm::dbgs() << "\n";
    });
    return true;
  }

  // If we arrive here, none of the patterns we tried matched.
  LLVM_DEBUG(llvm::dbgs() << "- Aborting: unknown clock scheme\n");
  return false;
}

/// Assuming there are two triggers, detect the clock and reset scheme
/// represented by a value table and store the results in `drive.reset` and
/// `drive.clock`.
bool Deseq::matchDriveClockAndReset(
    DriveInfo &drive,
    ArrayRef<std::pair<BetterDNFTerm, BetterValueEntry>> valueTable) {
  // We need exactly three entries in the value table to represent a register
  // with reset.
  if (valueTable.size() != 3) {
    LLVM_DEBUG(llvm::dbgs() << "- Aborting: two trigger value table has "
                            << valueTable.size() << " entries\n");
    return false;
  }

  // Resets take precedence over the clock, which shows up as `/rst` and
  // `/clk&rst` entries in the value table. We simply try all variant until we
  // find the one that fits.
  for (unsigned variant = 0; variant < (1 << 3); ++variant) {
    bool negClock = (variant >> 0) & 1;
    bool negReset = (variant >> 1) & 1;
    unsigned clockIdx = (variant >> 2) & 1;
    unsigned resetIdx = 1 - clockIdx;

    // Assemble the conditions in the value table corresponding to a clock edge
    // and reset edge, alongside the reset being active and inactive. The bit
    // patterns here follow from how we assign indices to past and present
    // triggers, and how the DNF's even bits represent positive terms and odd
    // bits represent inverted terms.
    uint32_t clockEdge = (negClock ? 0b1001 : 0b0110) << (clockIdx * 4 + 2);
    uint32_t resetEdge = (negReset ? 0b1001 : 0b0110) << (resetIdx * 4 + 2);
    uint32_t resetOn = (negReset ? 0b1000 : 0b0100) << (resetIdx * 4 + 2);
    uint32_t resetOff = (negReset ? 0b0100 : 0b1000) << (resetIdx * 4 + 2);

    // Combine the above bit masks into conditions for the reset edge, clock
    // edge with reset active, and clock edge with reset inactive and optional
    // enable condition.
    auto reset = BetterDNFTerm{resetEdge};
    auto clockWhileReset = BetterDNFTerm{clockEdge | resetOn};
    auto clockWithoutEnable = BetterDNFTerm{clockEdge | resetOff};
    auto clockWithEnable = BetterDNFTerm{clockEdge | resetOff | 0b01};

    // Find the entries corresponding to the above conditions.
    auto resetIt = llvm::find_if(
        valueTable, [&](auto &pair) { return pair.first == reset; });
    if (resetIt == valueTable.end())
      continue;

    auto clockWhileResetIt = llvm::find_if(
        valueTable, [&](auto &pair) { return pair.first == clockWhileReset; });
    if (clockWhileResetIt == valueTable.end())
      continue;

    auto clockIt = llvm::find_if(valueTable, [&](auto &pair) {
      return pair.first == clockWithoutEnable || pair.first == clockWithEnable;
    });
    if (clockIt == valueTable.end())
      continue;

    // Ensure that `/rst` and `/clk&rst` set the register to the same reset
    // value. Otherwise the reset doesn't have clear precedence over the
    // clock, and we can't turn this drive into a register.
    if (clockWhileResetIt->second != resetIt->second ||
        resetIt->second.isUnknown()) {
      LLVM_DEBUG(llvm::dbgs() << "- Aborting: inconsistent reset value\n");
      return false;
    }
    if (!isValidResetValue(resetIt->second.value)) {
      LLVM_DEBUG(llvm::dbgs() << "- Aborting: non-static reset value\n");
      return false;
    }

    // Populate the reset and clock info, and return.
    drive.reset.reset = observedValues[resetIdx];
    drive.reset.value = resetIt->second.value;
    drive.reset.activeHigh = !negReset;

    drive.clock.clock = observedValues[clockIdx];
    drive.clock.risingEdge = !negClock;
    if (clockIt->first == clockWithEnable)
      drive.clock.enable = drive.op.getEnable();
    drive.clock.value = drive.op.getValue();
    if (!clockIt->second.isUnknown())
      drive.clock.value = clockIt->second.value;

    LLVM_DEBUG({
      llvm::dbgs() << "  - Matched " << (negClock ? "neg" : "pos")
                   << "edge clock ";
      drive.clock.clock.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " -> " << clockIt->second;
      if (drive.clock.enable)
        llvm::dbgs() << " (with enable)";
      llvm::dbgs() << "\n";
      llvm::dbgs() << "  - Matched active-" << (negReset ? "low" : "high")
                   << " reset ";
      drive.reset.reset.printAsOperand(llvm::dbgs(), OpPrintingFlags());
      llvm::dbgs() << " -> " << resetIt->second << "\n";
    });
    return true;
  }

  // If we arrive here, none of the patterns we tried matched.
  LLVM_DEBUG(llvm::dbgs() << "- Aborting: unknown reset scheme\n");
  return false;
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
  if (!drive.clock.risingEdge)
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
          defOp && defOp->hasTrait<OpTrait::ConstantLike>())
        defOp->moveBefore(process);
      else
        resetValue = specializeValue(
            drive.op.getValue(),
            FixedValues{{drive.clock.clock, !drive.clock.risingEdge,
                         !drive.clock.risingEdge},
                        {drive.reset.reset, !drive.reset.activeHigh,
                         drive.reset.activeHigh}});
    }
  }

  // Determine the enable condition. If we have determined that the register
  // is trivially enabled, don't add an enable. If the enable condition is a
  // simple boolean value available outside the process, use it directly.
  Value enable = drive.clock.enable;
  if (enable && !enable.getParentRegion()->isProperAncestor(&process.getBody()))
    enable = drive.op.getEnable();

  // Determine the value. If the value is trivially available outside the
  // process, use it directly. If it is a constant, move the constant outside
  // the process.
  Value value = drive.clock.value;
  if (!value.getParentRegion()->isProperAncestor(&process.getBody())) {
    if (auto *defOp = value.getDefiningOp();
        defOp && defOp->hasTrait<OpTrait::ConstantLike>())
      defOp->moveBefore(process);
    else
      value = drive.op.getEnable();
  }

  // Specialize the process for the clock trigger, which will produce the
  // enable and the value for regular clock edges.
  FixedValues fixedValues;
  fixedValues.push_back(
      {drive.clock.clock, !drive.clock.risingEdge, drive.clock.risingEdge});
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
