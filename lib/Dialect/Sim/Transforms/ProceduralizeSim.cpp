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
#include "mlir/Transforms/RegionUtils.h"
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
  LogicalResult proceduralizeOps(Value clock, ArrayRef<Operation *> ops);
  void cleanup();

  // Mapping Clock -> List of simulation ops to proceduralize.
  SmallMapVector<Value, SmallVector<Operation *>, 2> simOpMap;

  // List of formatting ops to be pruned after proceduralization.
  SmallVector<Operation *> cleanupList;
};
} // namespace

LogicalResult
ProceduralizeSimPass::proceduralizeOps(Value clock, ArrayRef<Operation *> ops) {

  // List of uniqued values to become arguments of the TriggeredOp.
  SmallSetVector<Value, 4> arguments;
  // Map printf ops -> flattened list of fragments
  SmallDenseMap<PrintFormattedOp, SmallVector<Operation *>, 4> fragmentMap;
  SmallVector<Location> locs;
  SmallDenseSet<Value, 1> alwaysEnabledConditions;
  SmallSetVector<Value, 1> timeValues;
  SmallSetVector<Value, 1> getFileValues;
  SmallVector<Operation *> activeOps;

  locs.reserve(ops.size());

  auto collectValue = [&](auto &&self, Value value) -> void {
    if (!value)
      return;
    if (value.getDefiningOp<TimeOp>()) {
      timeValues.insert(value);
      return;
    }
    if (auto getFile = value.getDefiningOp<GetFileOp>()) {
      getFileValues.insert(value);
      for (auto fileNameOperand : getFile.getFileNameOperands())
        self(self, fileNameOperand);
      cleanupList.push_back(getFile);
      return;
    }
    arguments.insert(value);
  };

  for (auto *op : ops) {
    if (auto printOp = dyn_cast<PrintFormattedOp>(op)) {
      // Handle the print condition value. If it is not constant, it has to
      // become a region argument. If it is constant false, skip the operation.
      if (auto cstCond =
              printOp.getCondition().getDefiningOp<hw::ConstantOp>()) {
        if (cstCond.getValue().isAllOnes())
          alwaysEnabledConditions.insert(printOp.getCondition());
        else {
          printOp.erase();
          continue;
        }
      } else {
        arguments.insert(printOp.getCondition());
      }

      // Accumulate locations.
      locs.push_back(printOp.getLoc());

      // Get the flat list of formatting fragments and collect leaf fragments.
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
          printOp.emitError("Proceduralization of format strings passed as "
                            "block argument is unsupported.");
          return failure();
        }
        fragmentList.push_back(fmtOp);
        // For value formatters, the value to be formatted has to become
        // an argument.
        if (auto fmtVal = getFormattedValue(fmtOp)) {
          collectValue(collectValue, fmtVal);
          continue;
        }
        // Some formatting fragments do not have value operands.
        if (llvm::isa<FormatLiteralOp, FormatHierPathOp>(fmtOp))
          continue;
        printOp.emitError("Unsupported formatting fragment op in "
                          "proceduralization.")
                .attachNote(fmtOp->getLoc())
            << "unexpected fragment op is here";
        return failure();
      }

      collectValue(collectValue, printOp.getStream());
      activeOps.push_back(op);
      continue;
    }

    auto fflushOp = cast<FFlushOp>(op);
    if (auto cstCond =
            fflushOp.getCondition().getDefiningOp<hw::ConstantOp>()) {
      if (cstCond.getValue().isAllOnes())
        alwaysEnabledConditions.insert(fflushOp.getCondition());
      else {
        fflushOp.erase();
        continue;
      }
    } else {
      arguments.insert(fflushOp.getCondition());
    }

    locs.push_back(fflushOp.getLoc());
    collectValue(collectValue, fflushOp.getStream());
    activeOps.push_back(op);
  }

  if (activeOps.empty())
    return success();

  // Build the hw::TriggeredOp
  OpBuilder builder(activeOps.back());
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
  Operation *lastPreludeOp = nullptr;
  auto setPreludeInsertionPoint = [&]() {
    if (lastPreludeOp)
      builder.setInsertionPointAfter(lastPreludeOp);
    else
      builder.setInsertionPointToStart(trigOp.getBodyBlock());
  };

  for (auto timeValue : timeValues) {
    auto timeOp = timeValue.getDefiningOp<TimeOp>();
    assert(timeOp && "Expected sim.time defining op");
    auto *clonedTimeOp = builder.clone(*timeOp);
    lastPreludeOp = clonedTimeOp;
    argumentMapper.map(timeValue, clonedTimeOp->getResult(0));
  }

  for (auto getFileValue : getFileValues) {
    auto getFileOp = getFileValue.getDefiningOp<GetFileOp>();
    assert(getFileOp && "Expected sim.get_file defining op");
    setPreludeInsertionPoint();
    auto *clonedGetFileOp = builder.clone(*getFileOp, argumentMapper);
    lastPreludeOp = clonedGetFileOp;
    argumentMapper.map(getFileValue, clonedGetFileOp->getResult(0));
  }

  if (!alwaysEnabledConditions.empty()) {
    setPreludeInsertionPoint();
    auto cstTrue = builder.createOrFold<hw::ConstantOp>(
        fusedLoc, IntegerAttr::get(builder.getI1Type(), 1));
    if (auto *cstTrueOp = cstTrue.getDefiningOp())
      lastPreludeOp = cstTrueOp;
    for (auto cstCond : alwaysEnabledConditions)
      argumentMapper.map(cstCond, cstTrue);
  }

  SmallDenseMap<Operation *, Operation *> cloneMap;
  Value prevConditionValue;
  Block *prevConditionBlock = nullptr;
  auto getOrCreateConditionBlock = [&](Value cond, Location loc) {
    if (cond != prevConditionValue)
      prevConditionBlock = nullptr;

    if (!prevConditionBlock) {
      builder.setInsertionPointToEnd(trigOp.getBodyBlock());
      auto ifOp =
          mlir::scf::IfOp::create(builder, loc, TypeRange{}, cond, true, false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      mlir::scf::YieldOp::create(builder, loc);
      prevConditionBlock = builder.getBlock();
      prevConditionValue = cond;
    }
    return prevConditionBlock;
  };

  for (auto *op : activeOps) {
    if (auto printOp = dyn_cast<PrintFormattedOp>(op)) {
      // Create a copy of the required fragment operations within the
      // TriggeredOp's body.
      auto fragments = fragmentMap[printOp];
      SmallVector<Value> clonedOperands;
      setPreludeInsertionPoint();
      for (auto *fragment : fragments) {
        auto &fmtCloned = cloneMap[fragment];
        if (!fmtCloned) {
          fmtCloned = builder.clone(*fragment, argumentMapper);
          lastPreludeOp = fmtCloned;
        }
        clonedOperands.push_back(fmtCloned->getResult(0));
      }
      // Concatenate fragments to a single value if necessary.
      Value procPrintInput;
      if (clonedOperands.size() != 1)
        procPrintInput = builder.createOrFold<FormatStringConcatOp>(
            printOp.getLoc(), clonedOperands);
      else
        procPrintInput = clonedOperands.front();

      auto cond = argumentMapper.lookup(printOp.getCondition());
      auto *condBlock = getOrCreateConditionBlock(cond, printOp.getLoc());

      // Create the procedural print operation and prune the operations outside
      // of the TriggeredOp.
      builder.setInsertionPoint(condBlock->getTerminator());
      Value stream = argumentMapper.lookupOrDefault(printOp.getStream());
      PrintFormattedProcOp::create(builder, printOp.getLoc(), procPrintInput,
                                   stream, printOp.getUsePrintfCond());
      cleanupList.push_back(printOp.getInput().getDefiningOp());
      printOp.erase();
      continue;
    }

    auto fflushOp = cast<FFlushOp>(op);
    auto cond = argumentMapper.lookup(fflushOp.getCondition());
    auto *condBlock = getOrCreateConditionBlock(cond, fflushOp.getLoc());

    // Create the procedural fflush operation.
    builder.setInsertionPoint(condBlock->getTerminator());
    Value stream = argumentMapper.lookupOrDefault(fflushOp.getStream());
    FFlushProcOp::create(builder, fflushOp.getLoc(), stream);
    fflushOp.erase();
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
  simOpMap.clear();
  cleanupList.clear();

  auto theModule = getOperation();
  // Collect simulation operations grouped by their clock.
  theModule.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto printOp = dyn_cast<PrintFormattedOp>(op)) {
      simOpMap[printOp.getClock()].push_back(op);
      return;
    }
    if (auto fflushOp = dyn_cast<FFlushOp>(op))
      simOpMap[fflushOp.getClock()].push_back(op);
  });

  // Create a hw::TriggeredOp for each clock
  for (auto &[clock, ops] : simOpMap)
    if (failed(proceduralizeOps(clock, ops))) {
      signalPassFailure();
      return;
    }

  cleanup();

  mlir::IRRewriter rewriter(theModule);
  (void)mlir::runRegionDCE(rewriter, theModule->getRegions());
}
