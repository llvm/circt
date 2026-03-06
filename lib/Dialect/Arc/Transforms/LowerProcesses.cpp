//===- LowerProcesses.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LowerProcesses pass, which lowers LLHD process
// operations into Arc state allocations and control flow.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-processes"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERPROCESSES
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using namespace mlir;

namespace {

/// Information about a block in a process that needs an index.
struct BlockInfo {
  Block *block;
  unsigned index;
};

/// Analysis result for a process operation.
struct ProcessAnalysis {
  // Entry block (always gets index 0)
  Block *entryBlock;

  // Blocks that are targets of wait operations (get indices 1, 2, 3, ...)
  SmallVector<Block *> waitTargetBlocks;

  // Map from block to its assigned index
  DenseMap<Block *, unsigned> blockIndices;

  // Index for the halted state
  unsigned haltedIndex;

  // Result type of the process
  Type resultType;
};

/// Analyze a process and assign indices to blocks.
static FailureOr<ProcessAnalysis> analyzeProcess(llhd::ProcessOp processOp) {
  ProcessAnalysis analysis;

  // Get the entry block
  analysis.entryBlock = &processOp.getBody().front();
  analysis.blockIndices[analysis.entryBlock] = 0;

  // Collect all blocks that are targets of wait operations
  llvm::SetVector<Block *> waitTargets;
  for (Block &block : processOp.getBody()) {
    auto *terminator = block.getTerminator();

    // Check for llhd.wait
    if (auto waitOp = dyn_cast<llhd::WaitOp>(terminator)) {
      // Reject block operands on wait
      if (!waitOp.getDestOperands().empty()) {
        return processOp.emitError(
            "processes with block operands on llhd.wait are not supported");
      }

      // Add the destination block to wait targets
      Block *dest = waitOp.getDest();
      if (dest != analysis.entryBlock) {
        waitTargets.insert(dest);
      }
    }
    // llhd.halt is allowed
    else if (auto haltOp = dyn_cast<llhd::HaltOp>(terminator)) {
      // Nothing to do for halt
    }
    // Reject other terminators
    else {
      return processOp.emitError("unsupported terminator in process: ")
             << terminator->getName();
    }
  }

  // Assign indices to wait target blocks
  unsigned nextIndex = 1;
  for (Block *block : waitTargets) {
    analysis.blockIndices[block] = nextIndex;
    analysis.waitTargetBlocks.push_back(block);
    nextIndex++;
  }

  // Halted state gets the next index
  analysis.haltedIndex = nextIndex;

  // Get result type
  if (processOp.getNumResults() == 0) {
    return processOp.emitError("process must have exactly one result");
  }
  if (processOp.getNumResults() > 1) {
    return processOp.emitError(
        "processes with multiple results are not yet supported");
  }
  analysis.resultType = processOp.getResult(0).getType();

  return analysis;
}

struct LowerProcessesPass
    : public arc::impl::LowerProcessesBase<LowerProcessesPass> {
  void runOnOperation() override;
  LogicalResult lowerProcess(llhd::ProcessOp processOp, Value storageArg,
                             OpBuilder &allocBuilder,
                             ImplicitLocOpBuilder &bodyBuilder);
};

} // namespace

void LowerProcessesPass::runOnOperation() {
  auto module = getOperation();

  // Walk all arc.model operations to find processes
  SmallVector<std::pair<llhd::ProcessOp, arc::ModelOp>> processOps;
  module.walk([&](arc::ModelOp modelOp) {
    modelOp.walk([&](llhd::ProcessOp processOp) {
      processOps.push_back({processOp, modelOp});
    });
  });

  for (auto [processOp, modelOp] : processOps) {
    // Get the storage argument from the model
    Value storageArg = modelOp.getBodyBlock().getArgument(0);

    // Create builders for allocation and body
    OpBuilder allocBuilder(processOp);
    ImplicitLocOpBuilder bodyBuilder(processOp.getLoc(), processOp);

    if (failed(
            lowerProcess(processOp, storageArg, allocBuilder, bodyBuilder))) {
      signalPassFailure();
      return;
    }
  }
}

LogicalResult
LowerProcessesPass::lowerProcess(llhd::ProcessOp processOp, Value storageArg,
                                 OpBuilder &allocBuilder,
                                 ImplicitLocOpBuilder &bodyBuilder) {
  // Analyze the process
  auto analysisResult = analyzeProcess(processOp);
  if (failed(analysisResult))
    return failure();

  ProcessAnalysis &analysis = *analysisResult;

  LLVM_DEBUG(llvm::dbgs() << "Lowering process with "
                          << analysis.waitTargetBlocks.size()
                          << " wait target blocks\n");

  ImplicitLocOpBuilder builder(processOp.getLoc(), processOp);
  auto loc = processOp.getLoc();

  // Step 1: Allocate states
  // - resume_time: i64 (time in femtoseconds)
  // - resume_block: i16 (block index to resume at)
  // - proc_result: result type of the process

  auto i64Type = builder.getI64Type();
  auto i16Type = builder.getI16Type();

  allocBuilder.setInsertionPoint(processOp);
  auto resumeTimeState = arc::AllocStateOp::create(
      allocBuilder, loc, arc::StateType::get(i64Type), storageArg);
  auto resumeBlockState = arc::AllocStateOp::create(
      allocBuilder, loc, arc::StateType::get(i16Type), storageArg);
  auto procResultState = arc::AllocStateOp::create(
      allocBuilder, loc, arc::StateType::get(analysis.resultType), storageArg);

  // Step 2: Generate the control flow structure
  bodyBuilder.setInsertionPoint(processOp);

  // Read current time
  auto currentTime = arc::CurrentTimeOp::create(bodyBuilder, loc, storageArg);

  // Read resume time
  auto resumeTimeRead =
      arc::StateReadOp::create(bodyBuilder, loc, resumeTimeState);

  // Check if we should resume (current_time >= resume_time)
  auto shouldResume = comb::ICmpOp::create(
      bodyBuilder, loc, comb::ICmpPredicate::uge, currentTime, resumeTimeRead);

  // Create the scf.if for the resume check
  auto ifOp = scf::IfOp::create(bodyBuilder, loc, shouldResume, false);

  // Inside the if, create an scf.execute_region
  OpBuilder::InsertionGuard guard(bodyBuilder);
  bodyBuilder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  auto executeOp = scf::ExecuteRegionOp::create(bodyBuilder, loc, TypeRange{});
  Block *executeBlock = &executeOp.getRegion().emplaceBlock();
  Region &executeRegion = executeOp.getRegion();

  // Read the resume block index at the start of the execute region
  // Use a regular OpBuilder to avoid any ImplicitLocOpBuilder magic
  OpBuilder regularBuilder(bodyBuilder.getContext());
  regularBuilder.setInsertionPointToStart(executeBlock);

  // Create the state read with explicit location
  Value resumeBlockRead = arc::StateReadOp::create(
      regularBuilder, loc, resumeBlockState.getResult());

  // Create blocks for the switch statement within the execute region
  // We need: one block per indexed block + one halted block
  SmallVector<Block *> switchBlocks;
  SmallVector<APInt> caseValues;

  // Entry block (index 0)
  Block *entryBlock = bodyBuilder.createBlock(&executeRegion);
  switchBlocks.push_back(entryBlock);
  caseValues.push_back(APInt(16, 0));

  // Wait target blocks (indices 1, 2, 3, ...)
  for (unsigned i = 0; i < analysis.waitTargetBlocks.size(); ++i) {
    Block *block = bodyBuilder.createBlock(&executeRegion);
    switchBlocks.push_back(block);
    caseValues.push_back(APInt(16, i + 1));
  }

  // Halted block (used as default destination, not a case)
  Block *haltedBlock = bodyBuilder.createBlock(&executeRegion);

  // Now create the cf.switch after the state read in the entry block
  // We need to reset the insertion point because createBlock changes it!
  bodyBuilder.setInsertionPointToEnd(executeBlock);

  // Convert APInt case values to DenseIntElementsAttr
  auto caseValuesAttr = DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(caseValues.size())}, i16Type),
      caseValues);

  // Create empty operand lists for default and cases
  SmallVector<ValueRange> caseOperands(switchBlocks.size());

  // Create the switch with haltedBlock as default and switchBlocks as cases
  cf::SwitchOp::create(bodyBuilder, loc, resumeBlockRead, ValueRange{},
                       caseOperands, caseValuesAttr, haltedBlock, switchBlocks);

  // Fill in the switch blocks with the actual process logic
  // Map from original process blocks to their corresponding switch blocks
  IRMapping mapping;

  // Entry block (index 0) corresponds to the process entry block
  mapping.map(analysis.entryBlock, switchBlocks[0]);

  // Wait target blocks (indices 1, 2, 3, ...) correspond to their switch blocks
  for (unsigned i = 0; i < analysis.waitTargetBlocks.size(); ++i) {
    mapping.map(analysis.waitTargetBlocks[i], switchBlocks[i + 1]);
  }

  // Clone operations from process blocks into switch blocks
  for (Block &processBlock : processOp.getBody()) {
    // Find the corresponding switch block (if this block has an index)
    auto it = analysis.blockIndices.find(&processBlock);
    if (it == analysis.blockIndices.end()) {
      // This block doesn't have an index, so it's not a resumption point
      // We'll handle it when we clone the blocks that branch to it
      continue;
    }

    unsigned blockIndex = it->second;
    Block *switchBlock = switchBlocks[blockIndex];

    bodyBuilder.setInsertionPointToEnd(switchBlock);

    // Clone all operations except the terminator
    for (Operation &op : processBlock.without_terminator()) {
      bodyBuilder.clone(op, mapping);
    }

    // Handle the terminator specially
    auto *terminator = processBlock.getTerminator();
    if (auto waitOp = dyn_cast<llhd::WaitOp>(terminator)) {
      // Lower llhd.wait to state writes
      // 1. Write the yield value to proc_result state (if any)
      if (!waitOp.getYieldOperands().empty()) {
        Value yieldValue =
            mapping.lookupOrDefault(waitOp.getYieldOperands()[0]);
        arc::StateWriteOp::create(bodyBuilder, loc, procResultState.getResult(),
                                  yieldValue, Value{});
      }

      // 2. Get the target block index
      Block *destBlock = waitOp.getDest();
      auto destIt = analysis.blockIndices.find(destBlock);
      assert(destIt != analysis.blockIndices.end() &&
             "wait destination must have an index");
      unsigned destIndex = destIt->second;

      // 3. Calculate the resume time (current_time + delay)
      auto currentTime =
          arc::CurrentTimeOp::create(bodyBuilder, loc, storageArg);
      Value resumeTime = currentTime;

      if (waitOp.getDelay()) {
        // TODO: Properly convert llhd.time to i64 femtoseconds
        // For now, we'll just use current time (effectively no delay)
        // In a real implementation, we'd need to extract the time value
        // and convert it to femtoseconds
      }

      // 4. Write resume_time state
      arc::StateWriteOp::create(bodyBuilder, loc, resumeTimeState.getResult(),
                                resumeTime, Value{});

      // 5. Write resume_block state
      auto destIndexValue = arith::ConstantOp::create(
          bodyBuilder, loc, i16Type, bodyBuilder.getI16IntegerAttr(destIndex));
      arc::StateWriteOp::create(bodyBuilder, loc, resumeBlockState.getResult(),
                                destIndexValue, Value{});

      // 6. Yield from the execute region
      scf::YieldOp::create(bodyBuilder, loc);

    } else if (auto haltOp = dyn_cast<llhd::HaltOp>(terminator)) {
      // Lower llhd.halt to state writes
      // 1. Write the final value to proc_result state (if any)
      if (haltOp.getNumOperands() > 0) {
        Value finalValue = mapping.lookupOrDefault(haltOp.getOperand(0));
        arc::StateWriteOp::create(bodyBuilder, loc, procResultState.getResult(),
                                  finalValue, Value{});
      }

      // 2. Set resume_block to halted index
      auto haltedIndexValue = arith::ConstantOp::create(
          bodyBuilder, loc, i16Type,
          bodyBuilder.getI16IntegerAttr(analysis.haltedIndex));
      arc::StateWriteOp::create(bodyBuilder, loc, resumeBlockState.getResult(),
                                haltedIndexValue, Value{});

      // 3. Yield from the execute region
      scf::YieldOp::create(bodyBuilder, loc);

    } else {
      return processOp.emitError("unsupported terminator in process: ")
             << terminator->getName();
    }
  }

  // Fill in the halted block - just yield
  bodyBuilder.setInsertionPointToEnd(haltedBlock);
  scf::YieldOp::create(bodyBuilder, loc);

  // Replace the process result with a read from proc_result state
  bodyBuilder.setInsertionPointAfter(ifOp);
  auto procResultRead =
      arc::StateReadOp::create(bodyBuilder, loc, procResultState);
  processOp.getResult(0).replaceAllUsesWith(procResultRead);

  // Erase the original process
  processOp.erase();

  return success();
}

std::unique_ptr<Pass> arc::createLowerProcessesPass() {
  return std::make_unique<LowerProcessesPass>();
}
