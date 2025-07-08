//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeseqUtils.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

// Provide a `llhd-deseq` debug option for some high-level observability, and
// `llhd-deseq-verbose` for additional prints that trace out concrete values
// propagated across the IR.
#define DEBUG_TYPE "llhd-deseq"
#define VERBOSE_DEBUG(...) DEBUG_WITH_TYPE(DEBUG_TYPE "-verbose", __VA_ARGS__)

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_DESEQPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using namespace deseq;
using llvm::SmallSetVector;

namespace {
/// The work horse promoting processes into concrete registers.
struct Deseq {
  Deseq(ProcessOp process) : process(process) {}
  void deseq();

  bool analyzeProcess();
  Value tracePastValue(Value pastValue);

  TruthTable computeBoolean(Value value);
  ValueTable computeValue(Value value);
  TruthTable computeBoolean(OpResult value);
  ValueTable computeValue(OpResult value);
  TruthTable computeBoolean(BlockArgument value);
  ValueTable computeValue(BlockArgument arg);
  TruthTable computeBlockCondition(Block *block);
  TruthTable computeSuccessorCondition(BlockOperand &operand);
  TruthTable computeSuccessorBoolean(BlockOperand &operand, unsigned argIdx);
  ValueTable computeSuccessorValue(BlockOperand &operand, unsigned argIdx);

  bool matchDrives();
  bool matchDrive(DriveInfo &drive);
  bool matchDriveClock(DriveInfo &drive,
                       ArrayRef<std::pair<DNFTerm, ValueEntry>> valueTable);
  bool
  matchDriveClockAndReset(DriveInfo &drive,
                          ArrayRef<std::pair<DNFTerm, ValueEntry>> valueTable);

  void implementRegisters();
  void implementRegister(DriveInfo &drive);

  Value specializeValue(Value value, FixedValues fixedValues);
  ValueRange specializeProcess(FixedValues fixedValues);

  /// The process we are desequentializing.
  ProcessOp process;
  /// The single wait op of the process.
  WaitOp wait;
  /// The boolean values observed by the wait. These trigger the process and
  /// may cause the described register to update its value.
  SmallSetVector<Value, 2> triggers;
  /// The values carried from the past into the present as destination operands
  /// of the wait op. These values are guaranteed to also be contained in
  /// `triggers`.
  SmallVector<Value, 2> pastValues;
  /// The conditional drive operations fed by this process.
  SmallVector<DriveInfo> driveInfos;
  /// Specializations of the process for different trigger values.
  SmallDenseMap<FixedValues, ValueRange, 2> specializedProcesses;
  /// A cache of `seq.to_clock` ops.
  SmallDenseMap<Value, Value, 1> materializedClockCasts;
  /// A cache of `seq.clock_inv` ops.
  SmallDenseMap<Value, Value, 1> materializedClockInverters;
  /// A cache of `comb.xor` ops used as inverters.
  SmallDenseMap<Value, Value, 1> materializedInverters;
  /// An `llhd.constant_time` op created to represent an epsilon delay.
  ConstantTimeOp epsilonDelay;
  /// A map of operations that have been checked to be valid reset values.
  DenseMap<Operation *, bool> staticOps;

  /// The boolean expression computed for an `i1` value in the IR.
  DenseMap<Value, TruthTable> booleanLattice;
  /// The value table computed for a value in the IR. This essentially lists
  /// what values an SSA value assumes under certain conditions.
  DenseMap<Value, ValueTable> valueLattice;
  /// The condition under which control flow reaches a block. The block
  /// immediately following the wait op has this set to true; any further
  /// conditional branches will refine the condition of successor blocks.
  DenseMap<Block *, TruthTable> blockConditionLattice;
  /// The condition under which control flows along a terminator's block operand
  /// to its destination.
  DenseMap<BlockOperand *, TruthTable> successorConditionLattice;
  /// The boolean expression passed from a terminator to its destination as a
  /// destination block operand.
  DenseMap<std::pair<BlockOperand *, unsigned>, TruthTable>
      successorBooleanLattice;
  /// The value table passed from a terminator to its destination as a
  /// destination block operand.
  DenseMap<std::pair<BlockOperand *, unsigned>, ValueTable>
      successorValueLattice;

private:
  // Utilities to create boolean truth tables. These make working with truth
  // tables easier, since the calling code doesn't have to care about how
  // triggers and unknown value markers are packed into truth table columns.
  TruthTable getPoisonBoolean() const { return TruthTable::getPoison(); }
  TruthTable getUnknownBoolean() const {
    return TruthTable::getTerm(triggers.size() * 2 + 1, 0);
  }
  TruthTable getConstBoolean(bool value) const {
    return TruthTable::getConst(triggers.size() * 2 + 1, value);
  }
  TruthTable getPastTrigger(unsigned triggerIndex) const {
    return TruthTable::getTerm(triggers.size() * 2 + 1, triggerIndex * 2 + 1);
  }
  TruthTable getPresentTrigger(unsigned triggerIndex) const {
    return TruthTable::getTerm(triggers.size() * 2 + 1, triggerIndex * 2 + 2);
  }

  // Utilities to create value tables. These make working with value tables
  // easier, since the calling code doesn't have to care about how the truth
  // tables and value tables are constructed.
  ValueTable getUnknownValue() const {
    return ValueTable(getConstBoolean(true), ValueEntry::getUnknown());
  }
  ValueTable getPoisonValue() const {
    return ValueTable(getConstBoolean(true), ValueEntry::getPoison());
  }
  ValueTable getKnownValue(Value value) const {
    return ValueTable(getConstBoolean(true), value);
  }
};
} // namespace

/// Try to lower the process to a set of registers.
void Deseq::deseq() {
  // Check whether the process meets the basic criteria for being replaced by a
  // register. This includes having only a single `llhd.wait` op and feeding
  // only particular kinds of `llhd.drv` ops.
  if (!analyzeProcess())
    return;
  LLVM_DEBUG({
    llvm::dbgs() << "Desequentializing " << process.getLoc() << "\n";
    llvm::dbgs() << "- Feeds " << driveInfos.size() << " conditional drives\n";
    llvm::dbgs() << "- " << triggers.size() << " potential triggers:\n";
    for (auto [index, trigger] : llvm::enumerate(triggers)) {
      llvm::dbgs() << "  - ";
      trigger.printAsOperand(llvm::dbgs(), OpPrintingFlags());
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
    triggers.insert(value);
  }

  // We only support 1 or 2 observed values, since we map to registers with a
  // clock and an optional async reset.
  if (triggers.empty() || triggers.size() > 2) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc() << ": observes "
                            << triggers.size() << " values\n");
    return false;
  }

  // Seed the drive value analysis with the observed values.
  for (auto [index, trigger] : llvm::enumerate(triggers))
    booleanLattice.insert({trigger, getPresentTrigger(index)});

  // Ensure the wait op destination operands, i.e. the values passed from the
  // past into the present, are the observed values.
  for (auto [operand, blockArg] :
       llvm::zip(wait.getDestOperands(), wait.getDest()->getArguments())) {
    if (!operand.getType().isSignlessInteger(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                              << ": uses non-i1 past value\n");
      return false;
    }
    auto trigger = tracePastValue(operand);
    if (!trigger)
      return false;
    pastValues.push_back(trigger);
    unsigned index =
        std::distance(triggers.begin(), llvm::find(triggers, trigger));
    booleanLattice.insert({blockArg, getPastTrigger(index)});
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
    if (triggers.contains(value) || !arg) {
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
  if (!triggers.contains(distinctValue)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping " << process.getLoc()
                            << ": unobserved past value\n");
    return Value{};
  }
  return distinctValue;
}

//===----------------------------------------------------------------------===//
// Data Flow Analysis
//===----------------------------------------------------------------------===//

/// Convert a boolean SSA value into a truth table. If the value depends on any
/// of the process' triggers, that dependency is captured explicitly by the
/// truth table. Any other SSA values that factor into the value are represented
/// as an opaque term.
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
  if (auto it = booleanLattice.find(value); it != booleanLattice.end())
    return it->second;
  booleanLattice[value] = getUnknownBoolean();

  // Actually compute the value.
  TruthTable result =
      TypeSwitch<Value, TruthTable>(value).Case<OpResult, BlockArgument>(
          [&](auto value) { return computeBoolean(value); });

  // Memoize the result.
  VERBOSE_DEBUG({
    llvm::dbgs() << "- Boolean ";
    value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
    llvm::dbgs() << ": " << result << "\n";
  });
  booleanLattice[value] = result;
  return result;
}

/// Determine the different concrete values an SSA value may assume depending on
/// how control flow reaches the given value. This is used to determine the list
/// of different values that are driven onto a signal under various conditions.
ValueTable Deseq::computeValue(Value value) {
  // If this value is a result of the process we're analyzing, jump to the
  // corresponding yield operand of the wait op.
  if (value.getDefiningOp() == process)
    return computeValue(
        wait.getYieldOperands()[cast<OpResult>(value).getResultNumber()]);

  // Check if we have already computed this value. Otherwise insert an unknown
  // value to break recursions. This will be overwritten by a concrete value
  // later.
  if (auto it = valueLattice.find(value); it != valueLattice.end())
    return it->second;
  valueLattice[value] = getUnknownValue();

  // Actually compute the value.
  ValueTable result =
      TypeSwitch<Value, ValueTable>(value).Case<OpResult, BlockArgument>(
          [&](auto value) { return computeValue(value); });

  // Memoize the result.
  VERBOSE_DEBUG({
    llvm::dbgs() << "- Value ";
    value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
    llvm::dbgs() << ": " << result << "\n";
  });
  valueLattice[value] = result;
  return result;
}

/// Convert a boolean op result to a truth table.
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
        return result.isPoison() || (result != getUnknownBoolean() &&
                                     !result.isTrue() && !result.isFalse());
      }))
    return getPoisonBoolean();
  return getUnknownBoolean();
}

/// Determine the different values an op result may assume depending how control
/// flow reaches the op.
ValueTable Deseq::computeValue(OpResult value) {
  auto *op = value.getOwner();

  // Handle `comb.mux` and `arith.select`.
  if (isa<comb::MuxOp, arith::SelectOp>(op)) {
    auto condition = computeBoolean(op->getOperand(0));
    auto trueValue = computeValue(op->getOperand(1));
    auto falseValue = computeValue(op->getOperand(2));
    trueValue.addCondition(condition);
    falseValue.addCondition(~condition);
    trueValue.merge(std::move(falseValue));
    return trueValue;
  }

  // TODO: Reject values that depend on the triggers.
  return getKnownValue(value);
}

/// Convert a block argument to a truth table.
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

/// Determine the different values a block argument may assume depending how
/// control flow reaches the block.
ValueTable Deseq::computeValue(BlockArgument arg) {
  auto *block = arg.getOwner();

  // If this isn't a block in the process, simply return the value itself.
  if (block->getParentOp() != process)
    return getKnownValue(arg);

  // Otherwise iterate over all predecessors and compute the boolean values
  // being passed to this block argument by each.
  auto result = ValueTable();
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

/// Compute the boolean condition under which control flow reaches a block, as a
/// truth table.
TruthTable Deseq::computeBlockCondition(Block *block) {
  // Return a memoized result if one exists. Otherwise insert a default result
  // as recursion breaker.
  if (auto it = blockConditionLattice.find(block);
      it != blockConditionLattice.end())
    return it->second;
  blockConditionLattice[block] = getConstBoolean(false);

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
  VERBOSE_DEBUG({
    llvm::dbgs() << "- Block condition ";
    block->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ": " << result << "\n";
  });
  blockConditionLattice[block] = result;
  return result;
}

/// Compute the condition under which control transfers along a terminator's
/// block operand to the destination block.
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
  if (auto it = successorConditionLattice.find(&blockOperand);
      it != successorConditionLattice.end())
    return it->second;
  successorConditionLattice[&blockOperand] = getConstBoolean(false);

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
    result = getPoisonBoolean();
  }

  // Memoize the result.
  VERBOSE_DEBUG({
    llvm::dbgs() << "- Successor condition ";
    op->getBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#succ" << destIdx << " -> ";
    blockOperand.get()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << " = " << result << "\n";
  });
  successorConditionLattice[&blockOperand] = result;
  return result;
}

/// Compute the boolean value of a destination operand when control transfers
/// along a terminator's block operand to the destination block.
TruthTable Deseq::computeSuccessorBoolean(BlockOperand &blockOperand,
                                          unsigned argIdx) {
  // Return a memoized result if one exists. Otherwise insert a default result
  // as recursion breaker.
  if (auto it = successorBooleanLattice.find({&blockOperand, argIdx});
      it != successorBooleanLattice.end())
    return it->second;
  successorBooleanLattice[{&blockOperand, argIdx}] = getUnknownBoolean();

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
    result = getPoisonBoolean();
  }

  // Memoize the result.
  VERBOSE_DEBUG({
    llvm::dbgs() << "- Successor boolean ";
    op->getBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#succ" << destIdx << " -> ";
    blockOperand.get()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#arg" << argIdx << " = " << result << "\n";
  });
  successorBooleanLattice[{&blockOperand, argIdx}] = result;
  return result;
}

/// Determine the different values a destination operand may assume when control
/// transfers along a terminator's block operand to the destination block,
/// depending on how control flow reaches the terminator.
ValueTable Deseq::computeSuccessorValue(BlockOperand &blockOperand,
                                        unsigned argIdx) {
  // Return a memoized result if one exists. Otherwise insert a default result
  // as recursion breaker.
  if (auto it = successorValueLattice.find({&blockOperand, argIdx});
      it != successorValueLattice.end())
    return it->second;
  successorValueLattice[{&blockOperand, argIdx}] = getUnknownValue();

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
    result = getPoisonValue();
  }

  // Memoize the result.
  VERBOSE_DEBUG({
    llvm::dbgs() << "- Successor value ";
    op->getBlock()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#succ" << destIdx << " -> ";
    blockOperand.get()->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "#arg" << argIdx << " = " << result << "\n";
  });
  successorValueLattice[{&blockOperand, argIdx}] = result;
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
  SmallVector<std::pair<DNFTerm, ValueEntry>> valueTable;
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
  if (triggers.size() == 2)
    return matchDriveClockAndReset(drive, valueTable);

  // Otherwise we only have a single trigger, which is the clock.
  assert(triggers.size() == 1);
  return matchDriveClock(drive, valueTable);
}

/// Assuming there is one trigger, detect the clock scheme represented by a
/// value table and store the results in `drive.clock`.
bool Deseq::matchDriveClock(
    DriveInfo &drive, ArrayRef<std::pair<DNFTerm, ValueEntry>> valueTable) {
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
    auto clockWithoutEnable = DNFTerm{clockEdge};
    auto clockWithEnable = DNFTerm{clockEdge | 0b01};

    // Check if the single value table entry matches this clock.
    if (valueTable[0].first == clockWithEnable)
      drive.clock.enable = drive.op.getEnable();
    else if (valueTable[0].first != clockWithoutEnable)
      continue;

    // Populate the clock info and return.
    drive.clock.clock = triggers[0];
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
    DriveInfo &drive, ArrayRef<std::pair<DNFTerm, ValueEntry>> valueTable) {
  // We need exactly three entries in the value table to represent a register
  // with reset.
  if (valueTable.size() != 3) {
    LLVM_DEBUG(llvm::dbgs() << "- Aborting: two trigger value table has "
                            << valueTable.size() << " entries\n");
    return false;
  }

  // Resets take precedence over the clock, which shows up as `/rst` and
  // `/clk&rst` entries in the value table. We simply try all variants until we
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
    auto reset = DNFTerm{resetEdge};
    auto clockWhileReset = DNFTerm{clockEdge | resetOn};
    auto clockWithoutEnable = DNFTerm{clockEdge | resetOff};
    auto clockWithEnable = DNFTerm{clockEdge | resetOff | 0b01};

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

    // Populate the reset and clock info, and return.
    drive.reset.reset = triggers[resetIdx];
    drive.reset.value = resetIt->second.value;
    drive.reset.activeHigh = !negReset;

    drive.clock.clock = triggers[clockIdx];
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

//===----------------------------------------------------------------------===//
// Register Implementation
//===----------------------------------------------------------------------===//

/// Make all drives unconditional and implement the conditional behavior with
/// register ops.
void Deseq::implementRegisters() {
  for (auto &drive : driveInfos)
    implementRegister(drive);
}

/// Implement the conditional behavior of a drive with a `seq.firreg` op and
/// make the drive unconditional. This function pulls the analyzed clock and
/// reset from the given `DriveInfo` and creates the necessary ops outside the
/// process represent the behavior as a register. It also calls
/// `specializeValue` and `specializeProcess` to convert the sequential
/// `llhd.process` into a purely combinational `llhd.combinational` that is
/// simplified by assuming that the clock edge occurs.
void Deseq::implementRegister(DriveInfo &drive) {
  OpBuilder builder(drive.op);
  auto loc = drive.op.getLoc();

  // Materialize the clock as a `!seq.clock` value. Insert an inverter for
  // negedge clocks.
  auto &clockCast = materializedClockCasts[drive.clock.clock];
  if (!clockCast)
    clockCast = builder.create<seq::ToClockOp>(loc, drive.clock.clock);
  auto clock = clockCast;
  if (!drive.clock.risingEdge) {
    auto &clockInv = materializedClockInverters[clock];
    if (!clockInv)
      clockInv = builder.create<seq::ClockInverterOp>(loc, clock);
    clock = clockInv;
  }

  // Handle the optional reset.
  Value reset;
  Value resetValue;

  if (drive.reset) {
    reset = drive.reset.reset;
    resetValue = drive.reset.value;

    // Materialize the reset as an `i1` value. Insert an inverter for negedge
    // resets.
    if (!drive.reset.activeHigh) {
      auto &inv = materializedInverters[reset];
      if (!inv) {
        auto one = builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
        inv = builder.create<comb::XorOp>(loc, reset, one);
      }
      reset = inv;
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
      value = drive.op.getValue();
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

  // Try to guess a name for the register.
  StringAttr name;
  if (auto sigOp = drive.op.getSignal().getDefiningOp<llhd::SignalOp>())
    name = sigOp.getNameAttr();
  if (!name)
    name = builder.getStringAttr("");

  // Create the register op.
  auto reg =
      builder.create<seq::FirRegOp>(loc, value, clock, name, hw::InnerSymAttr{},
                                    /*preset=*/IntegerAttr{}, reset, resetValue,
                                    /*isAsync=*/reset != Value{});

  // If the register has an enable, insert a self-mux in front of the register.
  if (enable) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(reg);
    reg.getNextMutable().assign(builder.create<comb::MuxOp>(
        loc, enable, reg.getNext(), reg.getResult()));
  }

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
/// replace results of the process with results of a new combinational op. All
/// other behavior is purely an optimization; the function may not make use of
/// the assignments in `fixedValues` at all.
Value Deseq::specializeValue(Value value, FixedValues fixedValues) {
  auto result = dyn_cast<OpResult>(value);
  if (!result || result.getOwner() != process)
    return value;
  return specializeProcess(fixedValues)[result.getResultNumber()];
}

/// Specialize the current process by assuming the values listed in
/// `fixedValues` are at a constant value in the past and the present. This
/// function creates a new combinational op with a simplified version of the
/// process where all uses of the values listed in `fixedValues` are replaced
/// with their constant counterpart. Since the clock-dependent behavior of the
/// process has been absorbed into a register, the process can be replaced with
/// a combinational representation that computes the drive value and drive
/// condition under the assumption that the clock edge occurs.
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

  // Create an `llhd.combinational` op with this process specialized to compute
  // the result for the given fixed values. The triggers will be absorbed into
  // the register operation that consumes the result of this specialized
  // process, such that we can make the process purely combinational.
  OpBuilder builder(process);
  auto executeOp = builder.create<CombinationalOp>(process.getLoc(),
                                                   process.getResultTypes());

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

  // Compute the truth table that is true for the given fixed values, and false
  // otherwise. We will use that table to quickly evaluate booleans later.
  auto fixedTable = getConstBoolean(true);
  for (auto [index, value] : llvm::enumerate(triggers)) {
    for (auto fixedValue : fixedValues) {
      if (fixedValue.value != value)
        continue;
      auto past = getPastTrigger(index);
      fixedTable &= fixedValue.past ? past : ~past;
      auto present = getPresentTrigger(index);
      fixedTable &= fixedValue.present ? present : ~present;
      break;
    }
  }

  // Clone operations over.
  auto cloneBlocks = [&](bool stopAtWait) {
    SmallVector<Value> foldedResults;
    while (!worklist.empty()) {
      auto [oldBlock, newBlock] = worklist.pop_back_val();
      builder.setInsertionPointToEnd(newBlock);
      for (auto &oldOp : *oldBlock) {
        // Convert `llhd.wait` into `llhd.yield`.
        if (auto waitOp = dyn_cast<WaitOp>(oldOp)) {
          if (stopAtWait)
            continue;
          SmallVector<Value> operands;
          for (auto operand : waitOp.getYieldOperands())
            operands.push_back(mapping.lookupOrDefault(operand));
          builder.create<YieldOp>(waitOp.getLoc(), operands);
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

        // If our initial data flow analysis has produced a concrete boolean
        // value for an `i1`-valued op, see if it evaluates to a constant true
        // or false with the given fixed values.
        if (oldOp.getNumResults() == 1 &&
            oldOp.getResult(0).getType().isSignlessInteger(1)) {
          if (auto it = booleanLattice.find(oldOp.getResult(0));
              it != booleanLattice.end()) {
            if ((it->second & fixedTable).isFalse()) {
              mapping.map(oldOp.getResult(0), falseValue);
              continue;
            }
            if ((it->second & fixedTable) == fixedTable) {
              mapping.map(oldOp.getResult(0), trueValue);
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
