//===- ProceduralizeSim.cpp - Conversion to procedural operations ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform non-procedural simulation operations with clock and enable to
// procedural operations wrapped in a procedural region.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "proceduralize-sim"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_PROCEDURALIZESIM
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace llvm;
using namespace circt;
using namespace sim;

namespace {
struct ProceduralizeSimPass : impl::ProceduralizeSimBase<ProceduralizeSimPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult proceduralizePrintOps(Value clock,
                                      ArrayRef<PrintFormattedOp> printOps);
  SmallVector<Operation *> getPrintFragments(PrintFormattedOp op);
  void cleanup();

  // Mapping Clock -> List of printf ops
  SmallMapVector<Value, SmallVector<PrintFormattedOp>, 2> printfOpMap;

  // List of formatting ops to be pruned after proceduralization.
  SmallVector<Operation *> cleanupList;
};
} // namespace

LogicalResult ProceduralizeSimPass::proceduralizePrintOps(
    Value clock, ArrayRef<PrintFormattedOp> printOps) {

  // List of uniqued values to become arguments of the TriggeredOp.
  SmallSetVector<Value, 4> arguments;
  // Map printf ops -> flattened list of fragments
  SmallDenseMap<PrintFormattedOp, SmallVector<Operation *>, 4> fragmentMap;
  SmallVector<Location> locs;
  SmallDenseSet<Value, 1> alwaysEnabledConditions;

  locs.reserve(printOps.size());

  for (auto printOp : printOps) {
    // Handle the print condition value. If it is not constant, it has to become
    // a region argument. If it is constant false, skip the operation.
    if (auto cstCond = printOp.getCondition().getDefiningOp<hw::ConstantOp>()) {
      if (cstCond.getValue().isAllOnes())
        alwaysEnabledConditions.insert(printOp.getCondition());
      else
        continue;
    } else {
      arguments.insert(printOp.getCondition());
    }

    // Accumulate locations
    locs.push_back(printOp.getLoc());

    // Get the flat list of formatting fragments and collect leaf fragments
    SmallVector<Value> flatString;
    if (auto concatInput =
            printOp.getInput().getDefiningOp<FormatStringConcatOp>()) {

      auto isAcyclic = concatInput.getFlattenedInputs(flatString);
      if (failed(isAcyclic)) {
        printOp.emitError("Cyclic format string cannot be proceduralized.");
        return failure();
      }
    } else {
      flatString.push_back(printOp.getInput());
    }

    auto &fragmentList = fragmentMap[printOp];
    assert(fragmentList.empty() && "printf operation visited twice.");

    for (auto &fragment : flatString) {
      auto *fmtOp = fragment.getDefiningOp();
      if (!fmtOp) {
        printOp.emitError("Proceduralization of format strings passed as block "
                          "argument is unsupported.");
        return failure();
      }
      fragmentList.push_back(fmtOp);
      // For non-literal fragments, the value to be formatted has to become an
      // argument.
      if (!llvm::isa<FormatLiteralOp>(fmtOp)) {
        auto fmtVal = getFormattedValue(fmtOp);
        assert(!!fmtVal && "Unexpected foramtting fragment op.");
        arguments.insert(fmtVal);
      }
    }
  }

  // Build the hw::TriggeredOp
  OpBuilder builder(printOps.back());
  auto fusedLoc = builder.getFusedLoc(locs);

  SmallVector<Value> argVec = arguments.takeVector();

  auto clockConv = builder.createOrFold<seq::FromClockOp>(fusedLoc, clock);
  auto trigOp = hw::TriggeredOp::create(
      builder, fusedLoc,
      hw::EventControlAttr::get(builder.getContext(),
                                hw::EventControl::AtPosEdge),
      clockConv, argVec);

  // Map the collected arguments to the newly created block arguments.
  IRMapping argumentMapper;
  unsigned idx = 0;
  for (auto arg : argVec) {
    argumentMapper.map(arg, trigOp.getBodyBlock()->getArgument(idx));
    idx++;
  }

  // Materialize and map a 'true' constant within the TriggeredOp if required.
  builder.setInsertionPointToStart(trigOp.getBodyBlock());
  if (!alwaysEnabledConditions.empty()) {
    auto cstTrue = builder.createOrFold<hw::ConstantOp>(
        fusedLoc, IntegerAttr::get(builder.getI1Type(), 1));
    for (auto cstCond : alwaysEnabledConditions)
      argumentMapper.map(cstCond, cstTrue);
  }

  SmallDenseMap<Operation *, Operation *> cloneMap;
  Value prevConditionValue;
  Block *prevConditionBlock;

  for (auto printOp : printOps) {

    // Throw away disabled prints
    if (auto cstCond = printOp.getCondition().getDefiningOp<hw::ConstantOp>()) {
      if (cstCond.getValue().isZero()) {
        printOp.erase();
        continue;
      }
    }

    // Create a copy of the required fragment operations within the
    // TriggeredOp's body.
    auto fragments = fragmentMap[printOp];
    SmallVector<Value> clonedOperands;
    builder.setInsertionPointToStart(trigOp.getBodyBlock());
    for (auto *fragment : fragments) {
      auto &fmtCloned = cloneMap[fragment];
      if (!fmtCloned)
        fmtCloned = builder.clone(*fragment, argumentMapper);
      clonedOperands.push_back(fmtCloned->getResult(0));
    }
    // Concatenate fragments to a single value if necessary.
    Value procPrintInput;
    if (clonedOperands.size() != 1)
      procPrintInput = builder.createOrFold<FormatStringConcatOp>(
          printOp.getLoc(), clonedOperands);
    else
      procPrintInput = clonedOperands.front();

    // Check if we can reuse the previous conditional block.
    auto condArg = argumentMapper.lookup(printOp.getCondition());
    if (condArg != prevConditionValue)
      prevConditionBlock = nullptr;
    auto *condBlock = prevConditionBlock;

    // If not, create a new scf::IfOp for the condition.
    if (!condBlock) {
      builder.setInsertionPointToEnd(trigOp.getBodyBlock());
      auto ifOp = mlir::scf::IfOp::create(builder, printOp.getLoc(),
                                          TypeRange{}, condArg, true, false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      mlir::scf::YieldOp::create(builder, printOp.getLoc());
      condBlock = builder.getBlock();
      prevConditionValue = condArg;
      prevConditionBlock = condBlock;
    }

    // Create the procedural print operation and prune the operations outside of
    // the TriggeredOp.
    builder.setInsertionPoint(condBlock->getTerminator());
    PrintFormattedProcOp::create(builder, printOp.getLoc(), procPrintInput);
    cleanupList.push_back(printOp.getInput().getDefiningOp());
    printOp.erase();
  }
  return success();
}

// Prune the DAGs of formatting fragments left outside of the newly created
// TriggeredOps.
void ProceduralizeSimPass::cleanup() {
  SmallVector<Operation *> cleanupNextList;
  SmallDenseSet<Operation *> erasedOps;

  bool noChange = true;
  while (!cleanupList.empty() || !cleanupNextList.empty()) {

    if (cleanupList.empty()) {
      if (noChange)
        break;
      cleanupList = std::move(cleanupNextList);
      cleanupNextList = {};
      noChange = true;
    }

    auto *opToErase = cleanupList.pop_back_val();
    if (erasedOps.contains(opToErase))
      continue;

    if (opToErase->getUses().empty()) {
      // Remove a dead op. If it is a concat remove its operands, too.
      if (auto concat = dyn_cast<FormatStringConcatOp>(opToErase))
        for (auto operand : concat.getInputs())
          cleanupNextList.push_back(operand.getDefiningOp());
      opToErase->erase();
      erasedOps.insert(opToErase);
      noChange = false;
    } else {
      // Op still has uses, revisit later.
      cleanupNextList.push_back(opToErase);
    }
  }
}

void ProceduralizeSimPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");
  printfOpMap.clear();
  cleanupList.clear();

  auto theModule = getOperation();
  // Collect printf operations grouped by their clock.
  theModule.walk<mlir::WalkOrder::PreOrder>(
      [&](PrintFormattedOp op) { printfOpMap[op.getClock()].push_back(op); });

  // Create a hw::TriggeredOp for each clock
  for (auto &[clock, printOps] : printfOpMap)
    if (failed(proceduralizePrintOps(clock, printOps))) {
      signalPassFailure();
      return;
    }

  cleanup();
}
