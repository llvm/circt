//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-lower-processes"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_LOWERPROCESSESPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;

namespace {
struct Lowering {
  Lowering(ProcessOp processOp) : processOp(processOp) {}
  void lower();
  bool matchControlFlow();
  void markObservedValues();
  bool allOperandsObserved();
  bool isObserved(Value value);

  ProcessOp processOp;
  WaitOp waitOp;
  SmallDenseSet<Value> observedValues;
};
} // namespace

void Lowering::lower() {
  // Ensure that the process describes combinational logic.
  if (!matchControlFlow())
    return;
  markObservedValues();
  if (!allOperandsObserved())
    return;
  LLVM_DEBUG(llvm::dbgs() << "Lowering process " << processOp.getLoc() << "\n");

  // Replace the process.
  OpBuilder builder(processOp);
  auto executeOp = CombinationalOp::create(builder, processOp.getLoc(),
                                           processOp.getResultTypes());
  executeOp.getRegion().takeBody(processOp.getBody());
  processOp.replaceAllUsesWith(executeOp);
  processOp.erase();
  processOp = {};

  // Replace the `llhd.wait` with an `llhd.yield`.
  builder.setInsertionPoint(waitOp);
  YieldOp::create(builder, waitOp.getLoc(), waitOp.getYieldOperands());
  waitOp.erase();

  // Simplify the execute op body region since disconnecting the control flow
  // loop through the wait op has potentially created unreachable blocks.
  IRRewriter rewriter(builder);
  (void)simplifyRegions(rewriter, executeOp->getRegions());
}

/// Check that the process' entry block trivially joins a control flow loop
/// immediately after the wait op.
bool Lowering::matchControlFlow() {
  // Ensure that there is only a single wait op in the process and that it has
  // no destination operands.
  for (auto &block : processOp.getBody()) {
    if (auto op = dyn_cast<WaitOp>(block.getTerminator())) {
      if (waitOp) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                                << ": multiple wait ops\n");
        return false;
      }
      waitOp = op;
    }
  }
  if (!waitOp) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                            << ": no wait op\n");
    return false;
  }
  if (!waitOp.getDestOperands().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                            << ": wait op has destination operands\n");
    return false;
  }

  // Helper function to skip across empty blocks with only a single successor.
  auto skipToMergePoint = [&](Block *block) -> std::pair<Block *, ValueRange> {
    ValueRange operands;
    while (auto branchOp = dyn_cast<cf::BranchOp>(block->getTerminator())) {
      if (llvm::any_of(block->without_terminator(),
                       [](auto &op) { return !isMemoryEffectFree(&op); }))
        break;
      block = branchOp.getDest();
      operands = branchOp.getDestOperands();
      if (std::distance(block->pred_begin(), block->pred_end()) > 1)
        break;
      if (!operands.empty())
        break;
    }
    return {block, operands};
  };

  // Ensure that the entry block and wait op converge on the same block.
  auto &entry = processOp.getBody().front();
  auto [entryMergeBlock, entryMergeArgs] = skipToMergePoint(&entry);
  auto [waitMergeBlock, waitMergeArgs] = skipToMergePoint(waitOp.getDest());
  if (entryMergeBlock != waitMergeBlock) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping process " << processOp.getLoc()
               << ": control from entry and wait does not converge\n");
    return false;
  }

  // Helper function to check if two values are equivalent.
  auto areValuesEquivalent = [](std::tuple<Value, Value> values) {
    auto [a, b] = values;
    if (a == b)
      return true;
    auto *opA = a.getDefiningOp();
    auto *opB = b.getDefiningOp();
    if (!opA || !opB)
      return false;
    return OperationEquivalence::isEquivalentTo(
        opA, opB, OperationEquivalence::IgnoreLocations);
  };

  // Ensure that the entry block and wait op converge with equivalent block
  // arguments.
  if (!llvm::all_of(llvm::zip(entryMergeArgs, waitMergeArgs),
                    areValuesEquivalent)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                            << ": control from entry and wait converges with "
                               "different block arguments\n");
    return false;
  }

  // Ensure that no values are live across the wait op.
  Liveness liveness(processOp);
  for (auto value : liveness.getLiveOut(waitOp->getBlock())) {
    if (value.getParentRegion()->isProperAncestor(&processOp.getBody()))
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "Skipping process " << processOp.getLoc() << ": value ";
      value.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << " live across wait\n";
    });
    return false;
  }

  return true;
}

/// Mark values the process observes that are defined outside the process.
void Lowering::markObservedValues() {
  SmallVector<Value> worklist;
  auto markObserved = [&](Value value) {
    if (observedValues.insert(value).second)
      worklist.push_back(value);
  };

  for (auto value : waitOp.getObserved())
    if (value.getParentRegion()->isProperAncestor(&processOp.getBody()))
      markObserved(value);

  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    auto *op = value.getDefiningOp();
    if (!op)
      continue;

    // Look through probe ops to mark the probe signal as well, just in case
    // there may be multiple probes of the same signal.
    if (auto probeOp = dyn_cast<PrbOp>(op))
      markObserved(probeOp.getSignal());

    // Look through operations that simply reshape incoming values into an
    // aggregate form from which any changes remain apparent.
    if (isa<hw::ArrayCreateOp, hw::StructCreateOp, comb::ConcatOp,
            hw::BitcastOp>(op))
      for (auto operand : op->getOperands())
        markObserved(operand);
  }
}

/// Ensure that any value defined outside the process that is used inside the
/// process is derived entirely from an observed value.
bool Lowering::allOperandsObserved() {
  // Collect all ancestor regions such that we can easily check if a value is
  // defined outside the process.
  SmallPtrSet<Region *, 4> properAncestors;
  for (auto *region = processOp->getParentRegion(); region;
       region = region->getParentRegion())
    properAncestors.insert(region);

  // Walk all operations under the process and check each operand.
  auto walkResult = processOp.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      // We only care about values defined outside the process.
      if (!properAncestors.count(operand.getParentRegion()))
        continue;

      // If the value is observed, all is well.
      if (isObserved(operand))
        continue;

      // Otherwise complain and abort.
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping process " << processOp.getLoc()
                     << ": unobserved value ";
        operand.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
        llvm::dbgs() << "\n";
      });
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

/// Check if a value is observed by the wait op, or all its operands are only
/// derived from observed values.
bool Lowering::isObserved(Value value) {
  // Check if the value is trivially observed.
  if (observedValues.contains(value))
    return true;

  // Otherwise get the operation that defines it such that we can check if the
  // value is derived from purely observed values. If it isn't define by an op,
  // the value is unobserved.
  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  // Otherwise visit all ops in the fan-in cone and ensure that they are
  // observed. If any value is unobserved, immediately return false.
  SmallDenseSet<Operation *> seenOps;
  SmallVector<Operation *> worklist;
  seenOps.insert(defOp);
  worklist.push_back(defOp);
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();

    // Give up on ops with nested regions.
    if (op->getNumRegions() != 0)
      return false;

    // Otherwise check that all operands are observed. If we haven't seen an
    // operand before, and it is not a signal, add it to the worklist to be
    // checked.
    for (auto operand : op->getOperands()) {
      if (observedValues.contains(operand))
        continue;
      if (isa<hw::InOutType>(operand.getType()))
        return false;
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !isMemoryEffectFree(defOp))
        return false;
      if (seenOps.insert(defOp).second)
        worklist.push_back(defOp);
    }
  }

  // If we arrive at this point, we weren't able to reach an unobserved value.
  // Therefore we consider this value derived from only observed values.
  observedValues.insert(value);
  return true;
}

namespace {
struct LowerProcessesPass
    : public llhd::impl::LowerProcessesPassBase<LowerProcessesPass> {
  void runOnOperation() override;
};

static LogicalResult dropRedundantArguments(RewriterBase &rewriter,
                                            Block &block) {
  SmallVector<size_t> argsToErase;

  // Go through the arguments of the block.
  for (auto [argIdx, blockOperand] : llvm::enumerate(block.getArguments())) {
    bool sameArg = true;
    Value commonValue;

    // Go through the block predecessor and flag if they pass to the block
    // different values for the same argument.
    for (Block::pred_iterator predIt = block.pred_begin(),
                              predE = block.pred_end();
         predIt != predE; ++predIt) {
      Operation *terminator = (*predIt)->getTerminator();
      auto branchOperand =
          llvm::TypeSwitch<Operation *, std::optional<Value>>(terminator)
              .Case([&, argIdx = argIdx](BranchOpInterface branchOp) {
                auto succIndex = predIt.getSuccessorIndex();
                SuccessorOperands succOperands =
                    branchOp.getSuccessorOperands(succIndex);
                auto branchOperands = succOperands.getForwardedOperands();
                return branchOperands[argIdx];
              })
              .Case([&, argIdx = argIdx](WaitOp waitOp) {
                auto destOperands = waitOp.getDestOperands();
                return destOperands[argIdx];
              })
              .Default([](Operation *) { return std::nullopt; });

      if (!branchOperand.has_value()) {
        sameArg = false;
        break;
      }

      if (!commonValue) {
        commonValue = *branchOperand;
        continue;
      }

      if (*branchOperand != commonValue) {
        sameArg = false;
        break;
      }
    }

    // If they are passing the same value, drop the argument.
    if (commonValue && sameArg) {
      argsToErase.push_back(argIdx);

      // Remove the argument from the block.
      rewriter.replaceAllUsesWith(blockOperand, commonValue);
    }
  }

  // Remove the arguments.
  for (size_t argIdx : llvm::reverse(argsToErase)) {
    block.eraseArgument(argIdx);

    // Remove the argument from the branch ops.
    for (auto predIt = block.pred_begin(), predE = block.pred_end();
         predIt != predE; ++predIt) {
      llvm::TypeSwitch<Operation *, void>((*predIt)->getTerminator())
          .Case([&](BranchOpInterface branchOp) {
            auto branch = cast<BranchOpInterface>((*predIt)->getTerminator());
            unsigned succIndex = predIt.getSuccessorIndex();
            SuccessorOperands succOperands =
                branch.getSuccessorOperands(succIndex);
            succOperands.erase(argIdx);
          })
          .Case([&](WaitOp waitOp) {
            auto destOperands = waitOp.getDestOperandsMutable();
            destOperands.erase(argIdx);
          })
          .Default([](Operation *) {
            llvm_unreachable("Unexpected predecessor terminator");
          });
    }
  }

  return success(!argsToErase.empty());
}

static void simplifyProcess(ProcessOp &processOp) {
  OpBuilder builder(processOp);
  IRRewriter rewriter(builder);
  (void)simplifyRegions(rewriter, processOp->getRegions());

  // simplifyRegions does not prune the destination operands of the `llhd.wait`
  // operation because it does not implement BranchOpInterface, due to its
  // side-effect semantics. Implementing BranchOpInterface could allow other
  // optimizations to forward branch operands from the `llhd.wait` operation to
  // the destination block where execution resumes. However, this would be
  // invalid, as the SSA value might change across the wait operation.
  // Therefore, we manually prune the destination block arguments ourselves.
  for (auto &block : processOp.getBody()) {
    auto waitOp = dyn_cast<WaitOp>(block.getTerminator());
    if (!waitOp)
      continue;

    auto dstOperands = waitOp.getDestOperands();
    if (dstOperands.empty())
      continue;

    (void)dropRedundantArguments(rewriter, *waitOp.getDest());
  }
}

} // namespace

void LowerProcessesPass::runOnOperation() {
  SmallVector<ProcessOp> processOps(getOperation().getOps<ProcessOp>());
  for (auto processOp : processOps) {
    // simplifyProcess(processOp);
    Lowering(processOp).lower();
  }
}
