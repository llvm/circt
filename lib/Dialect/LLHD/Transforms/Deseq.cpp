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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

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

  /// Merge the values of another table into this table.
  void merge(ValueTable &&other);
  /// Add a condition to all entries in this table. Erases entries from the
  /// table that become trivially false.
  void addCondition(const DNF &condition);
};
} // namespace

void ValueTable::merge(ValueTable &&other) {
  // TODO: be more careful about this
  for (auto &&entry : other.entries)
    entries.push_back(entry);
  llvm::sort(entries);
}

void ValueTable::addCondition(const DNF &condition) {
  llvm::erase_if(entries, [&](auto &entry) {
    entry.condition &= condition;
    return entry.condition.isFalse();
  });
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
struct Deseq {
  Deseq(ProcessOp process) : process(process) {}
  void deseq();

  bool analyzeAndCheck();

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

  /// The process we are desequentializing.
  ProcessOp process;
  /// The single wait op of the process.
  WaitOp wait;
  /// The conditional drive operations fed by this process.
  SmallVector<DrvOp> finalDrives;
  /// The triggers that cause the process to update its results.
  SmallSetVector<Value, 2> triggers;

  DenseMap<Value, DNF> booleanValues;
  DenseMap<Block *, DNF> blockConditions;
  DenseMap<Value, ValueTable> valueTables;
};
} // namespace

void Deseq::deseq() {
  if (!analyzeAndCheck())
    return;
  LLVM_DEBUG({
    llvm::dbgs() << "Desequentializing " << process.getLoc() << "\n";
    llvm::dbgs() << "- Feeds " << finalDrives.size() << " conditional drives\n";
    llvm::dbgs() << "- " << triggers.size() << " potential triggers:\n";
    for (auto trigger : triggers)
      llvm::dbgs() << "  - " << trigger << "\n";
  });

  for (auto driveOp : finalDrives) {
    LLVM_DEBUG(llvm::dbgs() << "- Analyzing " << driveOp << "\n");
    auto condition = computeBoolean(driveOp.getEnable());
    LLVM_DEBUG(llvm::dbgs() << "  - Condition: " << condition << "\n");
    auto value = computeValue(driveOp.getValue());
    value.addCondition(condition);
    LLVM_DEBUG(llvm::dbgs() << "  - Value: " << value << "\n");
  }
}

bool Deseq::analyzeAndCheck() {
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

  // Determine the values that may be involved in edge trigger detection.
  SmallPtrSet<Value, 4> observed;
  observed.insert(wait.getObserved().begin(), wait.getObserved().end());

  for (auto pastValue : wait.getDestOperands()) {
    // We can only reason about boolean past values.
    if (!pastValue.getType().isSignlessInteger(1)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping " << process.getLoc()
                 << ": uses non-i1 past value: " << pastValue << "\n");
      return false;
    }

    // We can only reason about past values defined outside the process.
    if (process.getBody().isAncestor(pastValue.getParentRegion())) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                              << ": uses past value defined internally: "
                              << pastValue << "\n");
      return false;
    }

    // We can only reason about past values if they cause the process to resume
    // when they change.
    if (!observed.contains(pastValue)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping " << process.getLoc()
                 << ": past value not observed: " << pastValue << "\n");
      return false;
    }

    triggers.insert(pastValue);
  }
  if (triggers.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping " << process.getLoc() << ": no triggers\n");
    return false;
  }

  return true;
}

DNF Deseq::computeBoolean(Value value) {
  assert(value.getType().isSignlessInteger(1));
  if (auto it = booleanValues.find(value); it != booleanValues.end())
    return it->second;
  booleanValues.insert({value, DNF()}); // recursion blocker
  DNF dnf;
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
