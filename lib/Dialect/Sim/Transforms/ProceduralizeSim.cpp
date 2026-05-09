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
static LogicalResult collectFormatStringFragments(
    Value formatString, PrintFormattedOp anchorOp,
    SmallVectorImpl<Operation *> &fragmentList,
    SmallSetVector<Operation *, 8> &allFStringFragments,
    SmallSetVector<Value, 4> &arguments) {
  SmallVector<Value> flatString;
  if (auto concatInput = formatString.getDefiningOp<FormatStringConcatOp>()) {
    auto isAcyclic = concatInput.getFlattenedInputs(flatString);
    if (failed(isAcyclic)) {
      anchorOp.emitError("Cyclic format string cannot be proceduralized.");
      return failure();
    }
  } else {
    flatString.push_back(formatString);
  }

  assert(fragmentList.empty() && "format string visited twice");
  for (auto fragment : flatString) {
    auto *fmtOp = fragment.getDefiningOp();
    if (!fmtOp) {
      anchorOp.emitError("Proceduralization of format strings passed as block "
                         "argument is unsupported.");
      return failure();
    }
    fragmentList.push_back(fmtOp);
    allFStringFragments.insert(fmtOp);

    // For non-literal fragments, the value to be formatted has to become a
    // triggered region argument.
    if (!llvm::isa<FormatLiteralOp>(fmtOp)) {
      auto fmtVal = getFormattedValue(fmtOp);
      assert(!!fmtVal && "Unexpected formatting fragment op.");
      arguments.insert(fmtVal);
    }
  }
  return success();
}

static Value
rematerializeFormatStringFromFragments(ArrayRef<Operation *> fragments,
                                       OpBuilder &builder, IRMapping &mapping,
                                       Location loc) {
  SmallVector<Value> operands;
  operands.reserve(fragments.size());
  for (auto *fragment : fragments) {
    auto cloned = mapping.lookupOrNull(fragment->getResult(0));
    assert(cloned && "missing cloned fragment");
    operands.push_back(cloned);
  }
  if (operands.size() == 1)
    return operands.front();
  return builder.createOrFold<FormatStringConcatOp>(loc, operands);
}

static Block *getOrCreateConditionBlock(OpBuilder &builder,
                                        hw::TriggeredOp trigOp, Location loc,
                                        Value condition,
                                        Value &prevConditionValue,
                                        Block *&prevConditionBlock) {
  if (condition != prevConditionValue)
    prevConditionBlock = nullptr;

  if (prevConditionBlock)
    return prevConditionBlock;

  builder.setInsertionPointToEnd(trigOp.getBodyBlock());
  auto ifOp = mlir::scf::IfOp::create(builder, loc, TypeRange{}, condition,
                                      true, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::scf::YieldOp::create(builder, loc);
  prevConditionValue = condition;
  prevConditionBlock = builder.getBlock();
  return prevConditionBlock;
}

struct ProceduralizeSimPass : impl::ProceduralizeSimBase<ProceduralizeSimPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult proceduralizePrintOps(Value clock,
                                      ArrayRef<PrintFormattedOp> printOps);
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
  // Map print ops -> flattened list of format-string fragments.
  SmallDenseMap<PrintFormattedOp, SmallVector<Operation *>, 4> printFragmentMap;
  // Map get_file ops -> flattened list of filename format-string fragments.
  SmallDenseMap<GetFileOp, SmallVector<Operation *>, 4> fileNameFragmentMap;
  // All non-concat format-string fragment ops needed in the triggered body.
  SmallSetVector<Operation *, 8> allFStringFragments;
  // Keep get_file ops in first-use order.
  SmallSetVector<GetFileOp, 4> getFileOps;
  SmallVector<Location> locs;
  SmallDenseSet<Value, 1> alwaysEnabledConditions;
  SmallVector<PrintFormattedOp> livePrintOps;

  locs.reserve(printOps.size());
  for (auto printOp : printOps) {
    if (auto cstCond = printOp.getCondition().getDefiningOp<hw::ConstantOp>()) {
      if (cstCond.getValue().isZero()) {
        printOp.erase();
        continue;
      }
      if (cstCond.getValue().isAllOnes())
        alwaysEnabledConditions.insert(printOp.getCondition());
    } else {
      arguments.insert(printOp.getCondition());
    }

    livePrintOps.push_back(printOp);
    locs.push_back(printOp.getLoc());

    auto &printFragments = printFragmentMap[printOp];
    if (failed(::collectFormatStringFragments(printOp.getInput(), printOp,
                                              printFragments,
                                              allFStringFragments, arguments)))
      return failure();

    if (auto stream = printOp.getStream()) {
      auto getFileOp = stream.getDefiningOp<GetFileOp>();
      if (!getFileOp) {
        if (!stream.getDefiningOp())
          printOp.emitError("proceduralization requires stream to be produced "
                            "by sim.get_file, block arguments are unsupported");
        else
          printOp.emitError("proceduralization requires stream to be produced "
                            "by sim.get_file");
        return failure();
      }
      getFileOps.insert(getFileOp);
      auto &fileNameFragments = fileNameFragmentMap[getFileOp];
      if (fileNameFragments.empty() &&
          failed(::collectFormatStringFragments(
              getFileOp.getFileName(), printOp, fileNameFragments,
              allFStringFragments, arguments)))
        return failure();
    }
  }

  if (livePrintOps.empty())
    return success();

  OpBuilder builder(livePrintOps.back());
  auto fusedLoc = builder.getFusedLoc(locs);
  SmallVector<Value> argVec = arguments.takeVector();

  auto clockConv = builder.createOrFold<seq::FromClockOp>(fusedLoc, clock);
  auto trigOp = hw::TriggeredOp::create(
      builder, fusedLoc,
      hw::EventControlAttr::get(builder.getContext(),
                                hw::EventControl::AtPosEdge),
      clockConv, argVec);

  IRMapping mapping;
  for (auto [idx, arg] : llvm::enumerate(argVec))
    mapping.map(arg, trigOp.getBodyBlock()->getArgument(idx));

  builder.setInsertionPointToStart(trigOp.getBodyBlock());
  if (!alwaysEnabledConditions.empty()) {
    auto cstTrue = builder.createOrFold<hw::ConstantOp>(
        fusedLoc, IntegerAttr::get(builder.getI1Type(), 1));
    for (auto cstCond : alwaysEnabledConditions)
      mapping.map(cstCond, cstTrue);
  }

  for (auto *fragment : allFStringFragments) {
    auto original = fragment->getResult(0);
    if (mapping.lookupOrNull(original))
      continue;
    auto *cloned = builder.clone(*fragment, mapping);
    mapping.map(original, cloned->getResult(0));
  }

  for (auto getFileOp : getFileOps) {
    auto &fileNameFragments = fileNameFragmentMap[getFileOp];
    Value clonedFileName = ::rematerializeFormatStringFromFragments(
        fileNameFragments, builder, mapping, getFileOp.getLoc());

    auto clonedGetFile =
        GetFileOp::create(builder, getFileOp.getLoc(), clonedFileName);
    mapping.map(getFileOp.getResult(), clonedGetFile.getResult());

    cleanupList.push_back(getFileOp);
    cleanupList.push_back(getFileOp.getFileName().getDefiningOp());
  }

  // Materialize print inputs before creating any conditional blocks.
  // Whether to actually construct strings eagerly/lazily is left to lowering
  // backends.
  SmallDenseMap<PrintFormattedOp, Value> procPrintInputMap;
  // Insert after rematerialized fragments/get_file ops so operands dominate.
  builder.setInsertionPointToEnd(trigOp.getBodyBlock());
  for (auto printOp : livePrintOps) {
    auto &printFragments = printFragmentMap[printOp];
    procPrintInputMap[printOp] = ::rematerializeFormatStringFromFragments(
        printFragments, builder, mapping, printOp.getLoc());
  }

  Value prevConditionValue;
  Block *prevConditionBlock = nullptr;
  for (auto printOp : livePrintOps) {
    auto condArg = mapping.lookup(printOp.getCondition());
    auto *condBlock =
        ::getOrCreateConditionBlock(builder, trigOp, printOp.getLoc(), condArg,
                                    prevConditionValue, prevConditionBlock);

    builder.setInsertionPoint(condBlock->getTerminator());
    Value procPrintInput = procPrintInputMap[printOp];

    Value procPrintStream;
    if (auto stream = printOp.getStream()) {
      procPrintStream = mapping.lookupOrNull(stream);
      if (!procPrintStream) {
        printOp.emitError("proceduralization failed to rematerialize stream");
        return failure();
      }
    }

    PrintFormattedProcOp::create(builder, printOp.getLoc(), procPrintInput,
                                 procPrintStream);
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
