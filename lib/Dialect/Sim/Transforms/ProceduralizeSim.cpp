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

#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

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
  SmallVector<Operation *> getPrintTokens(PrintFormattedOp op);
  void cleanup();

  // Mapping Clock -> List of printf ops
  SmallMapVector<Value, SmallVector<PrintFormattedOp>, 2> printfOpMap;

  // List of formatting ops to be pruned after proceduralization.
  SmallVector<Operation *> cleanupList;
};
} // namespace

// Flatten a conatenated format string value to a list of tokens.
SmallVector<Operation *>
ProceduralizeSimPass::getPrintTokens(PrintFormattedOp op) {
  SmallMapVector<FormatStringConcatOp, unsigned, 4> concatStack;
  SmallVector<Operation *> tokens;

  auto *defOp = op.getInput().getDefiningOp();

  if (!defOp) {
    op.emitError("Format string token must not be a block argument.");
    return {};
  }

  if (isa<sim::FormatBinOp, sim::FormatHexOp, sim::FormatDecOp,
          sim::FormatCharOp, sim::FormatLitOp>(defOp)) {
    // Trivial case: Only a single token.
    tokens.push_back(defOp);
    return tokens;
  }

  concatStack.insert({llvm::cast<FormatStringConcatOp>(defOp), 0});
  while (!concatStack.empty()) {
    auto top = concatStack.back();
    auto currentConcat = top.first;
    unsigned operandIndex = top.second;

    // Iterate over concatenated operands
    while (operandIndex < currentConcat.getNumOperands()) {
      auto *nextDefOp = currentConcat.getOperand(operandIndex).getDefiningOp();
      if (!nextDefOp) {
        currentConcat.emitError(
            "Format string token must not be a block argument.");
        return {};
      }
      if (isa<sim::FormatBinOp, sim::FormatHexOp, sim::FormatDecOp,
              sim::FormatCharOp, sim::FormatLitOp>(nextDefOp)) {
        // Found a (leaf) token: Push it into the list, continue with the next
        // operand
        tokens.push_back(nextDefOp);
        operandIndex++;
      } else {
        // Concat of a concat: Save the next operand index to visit on the
        // stack and put the new concat on top.
        auto nextConcat = llvm::cast<FormatStringConcatOp>(nextDefOp);
        if (concatStack.contains(nextConcat)) {
          nextConcat.emitError("Cyclic format string concatenation detected.");
          return {};
        }
        concatStack[currentConcat] = operandIndex + 1;
        concatStack.insert({nextConcat, 0});
        break;
      }
    }
    // Pop the concat of the stack if we have visited all operands.
    if (operandIndex >= currentConcat.getNumOperands())
      concatStack.pop_back();
  }

  return tokens;
}

LogicalResult ProceduralizeSimPass::proceduralizePrintOps(
    Value clock, ArrayRef<PrintFormattedOp> printOps) {

  // List of uniqued values to become arguments of the TriggeredOp.
  SmallSetVector<Value, 4> arguments;
  // Map printf ops -> flattened list of tokens
  SmallDenseMap<PrintFormattedOp, SmallVector<Operation *>, 4> tokenMap;
  SmallVector<Location> locs;

  SmallDenseSet<Value, 1> alwaysEnabledConditions;

  tokenMap.reserve(printOps.size());
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

    // Get the flat list of formatting tokens and save it.
    auto substInsert = tokenMap.try_emplace(printOp, getPrintTokens(printOp));
    assert(substInsert.second && "printf operation visited twice.");
    if (substInsert.first->second.empty())
      return failure(); // Flattening of tokens failed.

    // For non-literal tokens, the value to be formatted has to become an
    // argument.
    for (auto &token : substInsert.first->second) {
      if (!isa<FormatLitOp>(token))
        arguments.insert(token->getOperand(0));
    }
  }

  // Build the hw::TriggeredOp
  OpBuilder builder(printOps.back());
  auto fusedLoc = builder.getFusedLoc(locs);

  SmallVector<Value> argVec = arguments.takeVector();

  auto clockConv = builder.createOrFold<seq::FromClockOp>(fusedLoc, clock);
  auto trigOp = builder.create<hw::TriggeredOp>(
      fusedLoc,
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
  std::pair<Value, Block *> prevCondition{Value(), nullptr};

  for (auto printOp : printOps) {

    // Throw away disabled prints
    if (auto cstCond = printOp.getCondition().getDefiningOp<hw::ConstantOp>()) {
      if (cstCond.getValue().isZero()) {
        printOp.erase();
        continue;
      }
    }

    auto condArg = argumentMapper.lookup(printOp.getCondition());

    // Create a copy of the required token operations within the TriggeredOp's
    // body.
    auto tokens = tokenMap[printOp];
    SmallVector<Value> clonedOperands;
    builder.setInsertionPointToStart(trigOp.getBodyBlock());
    for (auto *token : tokens) {
      auto &fmtCloned = cloneMap[token];
      if (!fmtCloned)
        fmtCloned = builder.clone(*token, argumentMapper);
      clonedOperands.push_back(fmtCloned->getResult(0));
    }

    // Check if we can reuse the previous conditional block.
    if (condArg != prevCondition.first)
      prevCondition.second = nullptr;
    auto *condBlock = prevCondition.second;

    // If not, create a new scf::IfOp for the condition.
    if (!condBlock) {
      builder.setInsertionPointToEnd(trigOp.getBodyBlock());
      auto ifOp = builder.create<mlir::scf::IfOp>(printOp.getLoc(), TypeRange{},
                                                  condArg, true, false);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      builder.create<mlir::scf::YieldOp>(printOp.getLoc());
      condBlock = builder.getBlock();
      prevCondition.first = condArg;
      prevCondition.second = condBlock;
    }

    // Create the procedural print operation and prune the operations outside of
    // the TriggeredOp.
    builder.setInsertionPoint(condBlock->getTerminator());
    builder.create<PrintFormattedProcOp>(printOp.getLoc(), clonedOperands);
    cleanupList.push_back(printOp.getInput().getDefiningOp());
    printOp.erase();
  }
  return success();
}

// Prune the DAGs of formatting tokens left outside of the newly created
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
