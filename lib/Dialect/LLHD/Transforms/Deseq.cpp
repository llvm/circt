//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeseqDNF.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-deseq"

// TODO:
// - [ ] Ensure unknown ops don't "swallow" past values; these must be failures.
//       Example: add(@v0, v1) -> v2 is illegal, must be -> null
// - [ ] Non-i1 values from the past must poison everything. Null ValueTable?
// - [ ] After trigger analysis, re-propagate with reduced set of valid
//       past/present values for edge detection. Should produce null values for
//       all unsavory uses of a value.

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_DESEQPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallDenseSet;
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

namespace {
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

struct ValueTable {
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
      valueAndId.first.print(os, OpPrintingFlags().skipRegions());
    });
  }
  os << ")";

  return os;
}

namespace {
struct SuccessorValue {
  DNF condition;
  SmallVector<DNF, 1> booleanArgs;
  SmallVector<ValueTable, 1> valueArgs;

  bool operator==(const SuccessorValue &other) const {
    return condition == other.condition && booleanArgs == other.booleanArgs &&
           valueArgs == other.valueArgs;
  }

  bool operator!=(const SuccessorValue &other) const {
    return !(*this == other);
  }
};

using SuccessorValues = SmallVector<SuccessorValue, 2>;

struct Deseq {
  Deseq(ProcessOp process) : process(process) {}
  void deseq();

  bool analyzeProcess();
  bool analyzeTriggers();
  bool matchDrives();
  bool matchDrive(DrvOp driveOp);

  DNF computeBoolean(Value value);
  DNF computeBoolean(BlockArgument blockArg);
  DNF computeBoolean(OpResult result);
  DNF computeBoolean(comb::OrOp op);
  DNF computeBoolean(comb::AndOp op);
  DNF computeBoolean(comb::XorOp op);

  DNF computeBranchBoolean(Operation *terminator, Block *dest,
                           unsigned argNumber);
  DNF computeBranchBoolean(cf::BranchOp op, Block *dest, unsigned argNumber);
  DNF computeBranchBoolean(cf::CondBranchOp op, Block *dest,
                           unsigned argNumber);
  DNF computeBranchBoolean(WaitOp op, Block *dest, unsigned argNumber);

  DNF computeBlockCondition(Block *block);
  DNF computeBranchCondition(Operation *terminator, Block *dest);
  DNF computeBranchCondition(cf::BranchOp op, Block *dest);
  DNF computeBranchCondition(cf::CondBranchOp op, Block *dest);
  DNF computeBranchCondition(WaitOp op, Block *dest);

  ValueTable computeValue(Value value);
  ValueTable computeValue(BlockArgument blockArg);
  ValueTable computeValue(OpResult result);
  ValueTable computeValue(Operation *terminator, Block *dest,
                          unsigned argNumber);
  ValueTable computeValue(cf::BranchOp op, Block *dest, unsigned argNumber);
  ValueTable computeValue(cf::CondBranchOp op, Block *dest, unsigned argNumber);

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

  /// The process we are desequentializing.
  ProcessOp process;
  /// The single wait op of the process.
  WaitOp wait;
  /// The boolean values observed by the wait.
  SmallSetVector<Value, 2> observedBooleans;
  /// The conditional drive operations fed by this process.
  SmallVector<DrvOp> finalDrives;
  /// The triggers that cause the process to update its results.
  SmallSetVector<Value, 2> triggers;

  DenseMap<Value, DNF> booleanValues;
  DenseMap<Block *, DNF> blockConditions;
  DenseMap<Value, ValueTable> valueTables;

  SmallSetVector<PointerUnion<Operation *, Block *>, 4> dirtyNodes;
  DenseMap<Value, DNF> booleanLattice;
  DenseMap<Value, ValueTable> valueLattice;
  DenseMap<Block *, DNF> blockConditionLattice;
  DenseMap<Block *, SuccessorValues> successorLattice;
};
} // namespace

void Deseq::deseq() {
  if (!analyzeProcess())
    return;
  LLVM_DEBUG({
    llvm::dbgs() << "Desequentializing " << process.getLoc() << "\n";
    llvm::dbgs() << "- Feeds " << finalDrives.size() << " conditional drives\n";
  });
  propagate();
  if (!analyzeTriggers())
    return;
  LLVM_DEBUG({
    llvm::dbgs() << "- " << triggers.size() << " potential triggers:\n";
    for (auto trigger : triggers)
      llvm::dbgs() << "  - " << trigger << "\n";
  });
  if (!matchDrives())
    return;
}

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

    finalDrives.push_back(driveOp);
  }

  return true;
}

/// After propagating values across the lattice, determine which values may be
/// involved in trigger detection.
bool Deseq::analyzeTriggers() {
  for (auto operand : wait.getDestOperands()) {
    // Only i1 values can be triggers.
    if (!operand.getType().isSignlessInteger(1)) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Ignoring " << operand.getType() << " trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << "\n";
      });
      continue;
    }

    // Dataflow analysis must have produced a single value for this past
    // operand.
    auto dnf = booleanLattice.lookup(operand);
    Value value;
    AndTerm::Use use;
    if (auto single = dnf.getSingleTerm()) {
      std::tie(value, use) = *single;
    } else {
      LLVM_DEBUG({
        llvm::dbgs() << "- Ignoring trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << " with multiple values " << dnf << "\n";
      });
      continue;
    }

    // We can only reason about past values defined outside the process.
    if (process.getBody().isAncestor(value.getParentRegion())) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Ignoring trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << " defined inside the process\n";
      });
      continue;
    }

    // We can only reason about past values if they cause the process to resume
    // when they change.
    if (!observedBooleans.contains(value)) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Ignoring unobserved trigger ";
        operand.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << "\n";
      });
      continue;
    }

    triggers.insert(value);
  }

  if (triggers.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "- Aborting: no triggers\n");
    return false;
  }
  return true;
}

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

bool Deseq::matchDrives() {
  for (auto driveOp : finalDrives)
    if (!matchDrive(driveOp))
      return false;
  return true;
}

bool Deseq::matchDrive(DrvOp driveOp) {
  LLVM_DEBUG(llvm::dbgs() << "- Analyzing " << driveOp << "\n");

  // Determine under which condition the drive is enabled.
  auto condition = booleanLattice.lookup(driveOp.getEnable());
  if (condition.isNull()) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Aborting: null condition on " << driveOp << "\n");
    return false;
  }

  // Determine which value is driven under which conditions.
  auto valueTable = valueLattice.lookup(driveOp.getValue());
  valueTable.addCondition(condition);
  LLVM_DEBUG({
    llvm::dbgs() << "  - Condition: " << condition << "\n";
    llvm::dbgs() << "  - Value: " << valueTable << "\n";
  });

  // Determine to which edges the drive is sensitive, and ensure that each value
  // is only triggered by a single edge.
  enum class Edge { None = 0, Pos, Neg };
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
  SmallDenseSet<unsigned, 2> resetEntries;
  struct ResetInfo {
    Value reset;
    Value value;
    bool activeHigh;
  };
  SmallPtrSet<Value, 2> resetTriggers;
  SmallVector<ResetInfo, 2> resetInfos;

  for (auto [value, edge] : edges) {
    auto resetEdge =
        edge == Edge::Pos ? AndTerm::posEdge(value) : AndTerm::negEdge(value);
    auto resetActive =
        AndTerm(value, 1 << (edge == Edge::Pos ? AndTerm::Id : AndTerm::NotId));
    auto resetInactive =
        AndTerm(value, 1 << (edge == Edge::Pos ? AndTerm::NotId : AndTerm::Id));

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

    // The reset value must be a constant. Technically also non-constants can be
    // reset values. However, since async resets are level sensitive (even
    // though Verilog describes them as edge sensitive), the reset value would
    // have to be part of the wait op's observed values. We don't check for
    // that.
    auto resetValue = valueTable.entries[singleResetEntry].value;
    auto *resetDefOp = resetValue.getDefiningOp();
    if (!resetDefOp || !resetDefOp->hasTrait<OpTrait::ConstantLike>())
      continue;

    // Ensure that all other triggers in the reset entry contain the active
    // reset value as an AND term.
    bool allGated =
        llvm::all_of(valueTable.entries[singleResetEntry].condition.orTerms,
                     [&](auto &orTerm) {
                       // Ignore the reset edge term.
                       if (orTerm == OrTerm(resetEdge))
                         return true;
                       // Otherwise the term must contain the active reset.
                       return orTerm.contains(resetActive);
                     });
    if (!allGated)
      continue;

    // Ensure that all other entries besides the reset entry are contain the
    // inactive reset value as an AND term.
    allGated = llvm::all_of(
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
    resetEntries.insert(singleResetEntry);
    resetInfos.push_back({value, resetValue, edge == Edge::Pos});
    resetTriggers.insert(value);
  }

  // Remove the resets from the edge triggers.
  for (auto &resetInfo : resetInfos)
    edges.erase(resetInfo.reset);

  // Remove the reset entries from the drive value table. These are listed
  // separately in `resetInfos`.
  unsigned readIdx = 0, writeIdx = 0;
  for (; readIdx < valueTable.entries.size(); ++readIdx)
    if (!resetEntries.contains(readIdx))
      valueTable.entries[writeIdx++] = std::move(valueTable.entries[readIdx]);
  valueTable.entries.resize(writeIdx);

  // Remove the resets from the conditions in the value table.
  for (auto &entry : valueTable.entries) {
    for (auto &orTerm : entry.condition.orTerms) {
      llvm::erase_if(orTerm.andTerms, [&](auto &andTerm) {
        return resetTriggers.contains(andTerm.value);
      });
    }
  }

  // Group the value table entries by the remaining triggers.
  // SmallDenseSet<Value, ValueTable, 2> clockValueTables;
  // for (auto &entry : valueTable.entries) {
  //   for (auto &orTerm : entry.condition.orTerms) {
  //     llvm::erase_if(orTerm.andTerms, [&](auto &andTerm){

  //     });
  //   }
  // }
  struct ClockInfo {
    Value clock;
    Edge edge;
    ValueTable valueTable;
  };
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

  LLVM_DEBUG({
    for (auto &resetInfo : resetInfos) {
      llvm::dbgs() << "  - Reset ";
      resetInfo.reset.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << " (active " << (resetInfo.activeHigh ? "high" : "low")
                   << "): ";
      resetInfo.value.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    }
    for (auto &clockInfo : clockInfos) {
      llvm::dbgs() << "  - Clock ";
      clockInfo.clock.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << " ("
                   << (clockInfo.edge == Edge::Pos ? "posedge" : "negedge")
                   << "): " << clockInfo.valueTable << "\n";
    }
  });

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
  ClockInfo singleClock = clockInfos[0];
  ResetInfo singleReset;
  if (!resetInfos.empty())
    singleReset = resetInfos[0];

  return true;
}

DNF Deseq::computeBoolean(Value value) {
  assert(value.getType().isSignlessInteger(1));
  if (auto it = booleanValues.find(value); it != booleanValues.end())
    return it->second;
  booleanValues.insert({value, DNF()}); // recursion blocker
  auto dnf = DNF();
  if (triggers.contains(value))
    dnf = DNF(value);
  else if (auto blockArg = dyn_cast<BlockArgument>(value))
    dnf = computeBoolean(blockArg);
  else
    dnf = computeBoolean(cast<OpResult>(value));
  // LLVM_DEBUG(llvm::dbgs() << "- Value " << value << ": " << dnf << "\n");
  booleanValues[value] = dnf;
  return dnf;
}

DNF Deseq::computeBoolean(BlockArgument blockArg) {
  auto *block = blockArg.getOwner();
  auto result = DNF(false);
  SmallPtrSet<Block *, 8> seen;
  for (auto *predecessor : block->getPredecessors()) {
    if (seen.insert(predecessor).second) {
      result |= computeBranchBoolean(predecessor->getTerminator(), block,
                                     blockArg.getArgNumber());
      if (result.isTrue())
        break;
    }
  }
  return result;
}

DNF Deseq::computeBoolean(OpResult result) {
  auto *op = result.getOwner();

  // Handle results of this process.
  if (op == process)
    return computeBoolean(wait.getYieldOperands()[result.getResultNumber()]);

  // Handle constants.
  APInt intValue;
  if (op->hasTrait<OpTrait::ConstantLike>() &&
      m_ConstantInt(&intValue).match(op)) {
    assert(intValue.getBitWidth() == 1);
    return DNF(intValue.isOne());
  }

  // Handle known ops.
  return TypeSwitch<Operation *, DNF>(op)
      .Case<comb::OrOp, comb::AndOp, comb::XorOp>(
          [&](auto op) { return computeBoolean(op); })
      .Default([&](auto) { return DNF(result); });
}

DNF Deseq::computeBoolean(comb::OrOp op) {
  auto result = DNF(false);
  for (auto operand : op.getInputs()) {
    result |= computeBoolean(operand);
    if (result.isTrue())
      break;
  }
  return result;
}

DNF Deseq::computeBoolean(comb::AndOp op) {
  auto result = DNF(true);
  for (auto operand : op.getInputs()) {
    result &= computeBoolean(operand);
    if (result.isFalse())
      break;
  }
  return result;
}

DNF Deseq::computeBoolean(comb::XorOp op) {
  auto result = DNF(false);
  for (auto operand : op.getInputs())
    result ^= computeBoolean(operand);
  return result;
}

DNF Deseq::computeBranchBoolean(Operation *terminator, Block *dest,
                                unsigned argNumber) {
  DNF result = TypeSwitch<Operation *, DNF>(terminator)
                   .Case<cf::BranchOp, cf::CondBranchOp, WaitOp>([&](auto op) {
                     return computeBranchBoolean(op, dest, argNumber);
                   });

  // Incorporate the conditions under which control flow reaches the terminator.
  if (!result.isFalse())
    result &= computeBlockCondition(terminator->getBlock());

  // LLVM_DEBUG(llvm::dbgs() << "- Branch value " << *terminator << " (arg #"
  //                         << argNumber << ", to " << dest << "): " << result
  //                         << "\n");
  return result;
}

DNF Deseq::computeBranchBoolean(cf::BranchOp op, Block *dest,
                                unsigned argNumber) {
  return computeBoolean(op.getDestOperands()[argNumber]);
}

DNF Deseq::computeBranchBoolean(cf::CondBranchOp op, Block *dest,
                                unsigned argNumber) {
  auto result = DNF(false);

  // Handle the case where the destination is our true branch.
  if (op.getTrueDest() == dest) {
    auto destValue = computeBoolean(op.getTrueDestOperands()[argNumber]);
    if (!destValue.isFalse()) {
      auto condition = computeBoolean(op.getCondition());
      destValue &= condition;
      result |= destValue;
    }
  }

  // Handle the case where the destination is our false branch.
  if (op.getFalseDest() == dest) {
    auto destValue = computeBoolean(op.getFalseDestOperands()[argNumber]);
    if (!destValue.isFalse()) {
      auto condition = computeBoolean(op.getCondition());
      condition.negate();
      destValue &= condition;
      result |= destValue;
    }
  }

  return result;
}

DNF Deseq::computeBranchBoolean(WaitOp op, Block *dest, unsigned argNumber) {
  auto operand = op.getDestOperands()[argNumber];
  assert(triggers.contains(operand));
  return DNF::withPastValue(operand);
}

/// Compute the condition under which control flow reaches a block.
DNF Deseq::computeBlockCondition(Block *block) {
  if (auto it = blockConditions.find(block); it != blockConditions.end())
    return it->second;
  blockConditions.insert({block, DNF()}); // recursion blocker
  SmallPtrSet<Block *, 8> seen;
  auto result = DNF(false);
  for (auto *predecessor : block->getPredecessors()) {
    if (seen.insert(predecessor).second) {
      result |= computeBranchCondition(predecessor->getTerminator(), block);
      if (result.isTrue())
        break;
    }
  }
  // LLVM_DEBUG(llvm::dbgs() << "- Block condition " << block << ": " << result
  //                         << "\n");
  blockConditions[block] = result;
  return result;
}

DNF Deseq::computeBranchCondition(Operation *terminator, Block *dest) {
  DNF dnf = TypeSwitch<Operation *, DNF>(terminator)
                .Case<cf::BranchOp, cf::CondBranchOp, WaitOp>(
                    [&](auto op) { return computeBranchCondition(op, dest); });
  // LLVM_DEBUG(llvm::dbgs() << "- Branch condition " << *terminator << " (to "
  //                         << dest << "): " << dnf << "\n");
  return dnf;
}

DNF Deseq::computeBranchCondition(cf::BranchOp op, Block *dest) {
  return computeBlockCondition(op->getBlock());
}

DNF Deseq::computeBranchCondition(cf::CondBranchOp op, Block *dest) {
  auto result = computeBlockCondition(op->getBlock());
  if (result.isFalse())
    return result;

  // Handle the case where both branches go to the same destination.
  if (op.getTrueDest() == dest && op.getFalseDest() == dest)
    return result;

  // Handle the case where the destination is one of our branches.
  if (op.getTrueDest() == dest) {
    result &= computeBoolean(op.getCondition());
  } else {
    assert(op.getFalseDest() == dest);
    auto condition = computeBoolean(op.getCondition());
    condition.negate();
    result &= condition;
  }

  return result;
}

DNF Deseq::computeBranchCondition(WaitOp op, Block *dest) { return DNF(true); }

//===----------------------------------------------------------------------===//
// Value Tracing
//===----------------------------------------------------------------------===//

ValueTable Deseq::computeValue(Value value) {
  if (auto it = valueTables.find(value); it != valueTables.end())
    return it->second;
  valueTables.insert({value, ValueTable()}); // recursion blocker
  ValueTable table;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    table = computeValue(blockArg);
  else
    table = computeValue(cast<OpResult>(value));
  valueTables[value] = table;
  return table;
}

ValueTable Deseq::computeValue(BlockArgument blockArg) {
  auto *block = blockArg.getOwner();
  auto result = ValueTable();
  SmallPtrSet<Block *, 8> seen;
  for (auto *predecessor : block->getPredecessors()) {
    if (seen.insert(predecessor).second) {
      result.merge(computeValue(predecessor->getTerminator(), block,
                                blockArg.getArgNumber()));
    }
  }
  return result;
}

ValueTable Deseq::computeValue(OpResult result) {
  auto *op = result.getOwner();

  // Handle results of this process.
  if (op == process)
    return computeValue(wait.getYieldOperands()[result.getResultNumber()]);

  // Handle known ops.
  return TypeSwitch<Operation *, ValueTable>(op)
      // .Case<comb::OrOp, comb::AndOp, comb::XorOp>(
      //     [&](auto op) { return computeBoolean(op); })
      .Default([&](auto) { return ValueTable(result); });
}

ValueTable Deseq::computeValue(Operation *terminator, Block *dest,
                               unsigned argNumber) {
  ValueTable result = TypeSwitch<Operation *, ValueTable>(terminator)
                          .Case<cf::BranchOp, cf::CondBranchOp>([&](auto op) {
                            return computeValue(op, dest, argNumber);
                          });

  // Incorporate the conditions under which control flow reaches the terminator.
  if (!result.isEmpty())
    result.addCondition(computeBlockCondition(terminator->getBlock()));
  return result;
}

ValueTable Deseq::computeValue(cf::BranchOp op, Block *dest,
                               unsigned argNumber) {
  return computeValue(op.getDestOperands()[argNumber]);
}

ValueTable Deseq::computeValue(cf::CondBranchOp op, Block *dest,
                               unsigned argNumber) {
  auto result = ValueTable();

  // Handle the case where the destination is our true branch.
  if (op.getTrueDest() == dest) {
    auto destValue = computeValue(op.getTrueDestOperands()[argNumber]);
    if (!destValue.isEmpty()) {
      auto condition = computeBoolean(op.getCondition());
      destValue.addCondition(condition);
      result.merge(std::move(destValue));
    }
  }

  // Handle the case where the destination is our false branch.
  if (op.getFalseDest() == dest) {
    auto destValue = computeValue(op.getFalseDestOperands()[argNumber]);
    if (!destValue.isEmpty()) {
      auto condition = computeBoolean(op.getCondition());
      condition.negate();
      destValue.addCondition(condition);
      result.merge(std::move(destValue));
    }
  }

  return result;
}

void Deseq::markDirty(Block *block) { dirtyNodes.insert(block); }

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
    // LLVM_DEBUG({
    //   llvm::dbgs() << "- Update ";
    //   value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
    //   llvm::dbgs() << " = " << dnf << "\n";
    // });
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
    // LLVM_DEBUG({
    //   llvm::dbgs() << "- Update ";
    //   value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
    //   llvm::dbgs() << " = " << table << "\n";
    // });
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
    // LLVM_DEBUG({
    //   llvm::dbgs() << "- Update ";
    //   block->printAsOperand(llvm::dbgs());
    //   llvm::dbgs() << " successor " << index << " (";
    //   successor->printAsOperand(llvm::dbgs());
    //   llvm::dbgs() << ")\n";
    //   llvm::dbgs() << "  - Condition = " << values[index].condition << "\n";
    //   for (auto [argIdx, arg] : llvm::enumerate(values[index].arguments)) {
    //     llvm::dbgs() << "  - Arg ";
    //     successor->getArgument(argIdx).printAsOperand(llvm::dbgs(),
    //                                                   OpPrintingFlags());
    //     llvm::dbgs() << " = " << arg << "\n";
    //   }
    // });
    markDirty(successor);
  }
  slot = values;
}

/// Update the condition lattice value of a block, and mark all dependent nodes
/// as dirty.
void Deseq::updateCondition(Block *block, DNF condition) {
  auto &slot = blockConditionLattice[block];
  if (slot != condition) {
    // LLVM_DEBUG({
    //   llvm::dbgs() << "- Update ";
    //   block->printAsOperand(llvm::dbgs());
    //   llvm::dbgs() << " condition = " << condition << "\n";
    // });
    slot = condition;
    markDirty(block->getTerminator());
  }
}

void Deseq::propagate(Block *block) {
  // LLVM_DEBUG({
  //   llvm::dbgs() << "- Computing condition of ";
  //   block->printAsOperand(llvm::dbgs());
  //   llvm::dbgs() << "\n";
  // });

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
      // LLVM_DEBUG({
      //   llvm::dbgs() << "  - from ";
      //   predecessor->printAsOperand(llvm::dbgs());
      //   llvm::dbgs() << " successor " << blockOperand.getOperandNumber() <<
      //   ": "
      //                << successorValue.condition << "\n";
      //   llvm::dbgs() << "  - now " << condition << "\n";
      // });
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
            comb::XorOp>([&](auto op) { propagate(op); })
      .Case<llhd::PrbOp>([&](auto op) {
        if (op.getType().isSignlessInteger(1))
          updateBoolean(op, DNF(op.getResult()));
        updateValue(op, ValueTable(op.getResult()));
      });
}

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

void Deseq::propagate(comb::OrOp op) {
  auto result = DNF(false);
  for (auto operand : op.getInputs()) {
    result |= booleanLattice.lookup(operand);
    if (result.isTrue())
      break;
  }
  // if (result.isNull())
  //   result = DNF(op.getResult());
  updateBoolean(op, result);
}

void Deseq::propagate(comb::AndOp op) {
  auto result = DNF(true);
  for (auto operand : op.getInputs()) {
    result &= booleanLattice.lookup(operand);
    if (result.isFalse())
      break;
  }
  // if (result.isNull())
  //   result = DNF(op.getResult());
  updateBoolean(op, result);
}

void Deseq::propagate(comb::XorOp op) {
  auto result = DNF(false);
  for (auto operand : op.getInputs())
    result ^= booleanLattice.lookup(operand);
  // if (result.isNull())
  //   result = DNF(op.getResult());
  updateBoolean(op, result);
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
