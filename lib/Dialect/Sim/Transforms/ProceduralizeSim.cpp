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
#include "circt/Dialect/Sim/SimTransforms.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
static InFlightDiagnostic
emitProceduralizationError(const PrintProceduralizationRequest &request,
                           const Twine &message) {
  if (auto *anchor = request.anchorOp)
    return anchor->emitError(message);
  return mlir::emitError(request.loc, message);
}

static LogicalResult collectFormatStringFragments(
    Value formatString, const PrintProceduralizationRequest &request,
    SmallVectorImpl<Operation *> &fragmentList,
    SmallSetVector<Operation *, 8> &allFStringFragments,
    SmallSetVector<Value, 4> &arguments) {
  SmallVector<Value> flatString;
  if (auto concatInput = formatString.getDefiningOp<FormatStringConcatOp>()) {
    auto isAcyclic = concatInput.getFlattenedInputs(flatString);
    if (failed(isAcyclic)) {
      emitProceduralizationError(request, "Cyclic format string cannot be "
                                          "proceduralized.");
      return failure();
    }
  } else {
    flatString.push_back(formatString);
  }

  assert(fragmentList.empty() && "format string visited twice");
  for (auto fragment : flatString) {
    auto *fmtOp = fragment.getDefiningOp();
    if (!fmtOp) {
      emitProceduralizationError(
          request, "Proceduralization of format strings passed as block "
                   "argument is unsupported.");
      return failure();
    }
    fragmentList.push_back(fmtOp);
    allFStringFragments.insert(fmtOp);

    // Value formatter fragments have to carry their formatted value into the
    // triggered region. Contextual fragments such as sim.fmt.time or
    // sim.fmt.hier_path do not require an extra argument.
    if (auto fmtVal = getFormattedValue(fmtOp)) {
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

// Prune the DAGs of formatting fragments left outside of the newly created
// TriggeredOps.
static void cleanupDeadPrintArtifacts(ArrayRef<Operation *> cleanupList) {
  SmallVector<Operation *> worklist(cleanupList.begin(), cleanupList.end());
  SmallVector<Operation *> deferred;
  SmallDenseSet<Operation *> erasedOps;

  bool noChange = true;
  while (!worklist.empty() || !deferred.empty()) {
    if (worklist.empty()) {
      if (noChange)
        break;
      worklist = std::move(deferred);
      deferred.clear();
      noChange = true;
    }

    auto *opToErase = worklist.pop_back_val();
    if (!opToErase || erasedOps.contains(opToErase))
      continue;

    if (opToErase->getUses().empty()) {
      if (auto concat = dyn_cast<FormatStringConcatOp>(opToErase))
        for (auto operand : concat.getInputs())
          deferred.push_back(operand.getDefiningOp());
      opToErase->erase();
      erasedOps.insert(opToErase);
      noChange = false;
    } else {
      deferred.push_back(opToErase);
    }
  }
}

struct ProceduralizeSimPass : impl::ProceduralizeSimBase<ProceduralizeSimPass> {
public:
  void runOnOperation() override;

private:
  // Mapping Clock -> List of print requests.
  SmallMapVector<Value, SmallVector<PrintProceduralizationRequest>, 2>
      printRequestMap;
};
} // namespace

LogicalResult circt::sim::proceduralizePrintsForClock(
    OpBuilder &builder, Value clock,
    ArrayRef<PrintProceduralizationRequest> printRequests) {
  // List of uniqued values to become arguments of the TriggeredOp.
  SmallSetVector<Value, 4> arguments;
  // Map print requests -> flattened list of format-string fragments.
  SmallDenseMap<const PrintProceduralizationRequest *, SmallVector<Operation *>,
                4>
      printFragmentMap;
  // Map get_file ops -> flattened list of filename format-string fragments.
  SmallDenseMap<GetFileOp, SmallVector<Operation *>, 4> fileNameFragmentMap;
  // All non-concat format-string fragment ops needed in the triggered body.
  SmallSetVector<Operation *, 8> allFStringFragments;
  // Keep get_file ops in first-use order.
  SmallSetVector<GetFileOp, 4> getFileOps;
  SmallDenseSet<Value, 1> alwaysEnabledConditions;
  SmallVector<const PrintProceduralizationRequest *> livePrintRequests;
  SmallVector<Location> locs;
  SmallVector<Operation *> cleanupList;
  SmallVector<Operation *> sourceOpsToErase;

  locs.reserve(printRequests.size());
  for (const auto &request : printRequests) {
    if (auto cstCond = request.condition.getDefiningOp<hw::ConstantOp>()) {
      if (cstCond.getValue().isZero()) {
        if (auto *inputDef = request.input.getDefiningOp())
          cleanupList.push_back(inputDef);
        if (auto stream = request.stream) {
          if (auto getFileOp = stream.getDefiningOp<GetFileOp>()) {
            cleanupList.push_back(getFileOp);
            cleanupList.push_back(getFileOp.getFileName().getDefiningOp());
          } else if (auto *streamDef = stream.getDefiningOp()) {
            cleanupList.push_back(streamDef);
          }
        }
        if (request.cleanupRoot)
          sourceOpsToErase.push_back(request.cleanupRoot);
        continue;
      }
      if (cstCond.getValue().isAllOnes()) {
        alwaysEnabledConditions.insert(request.condition);
      } else {
        arguments.insert(request.condition);
      }
    } else {
      arguments.insert(request.condition);
    }

    livePrintRequests.push_back(&request);
    locs.push_back(request.loc);

    auto &printFragments = printFragmentMap[&request];
    if (failed(collectFormatStringFragments(request.input, request,
                                            printFragments, allFStringFragments,
                                            arguments)))
      return failure();

    if (auto stream = request.stream) {
      if (auto getFileOp = stream.getDefiningOp<GetFileOp>()) {
        getFileOps.insert(getFileOp);
        auto &fileNameFragments = fileNameFragmentMap[getFileOp];
        if (fileNameFragments.empty() &&
            failed(collectFormatStringFragments(
                getFileOp.getFileName(), request, fileNameFragments,
                allFStringFragments, arguments)))
          return failure();
      } else {
        if (!stream.getDefiningOp()) {
          emitProceduralizationError(
              request, "proceduralization requires stream to be produced by "
                       "sim.get_file, block arguments are unsupported");
        } else {
          emitProceduralizationError(request,
                                     "proceduralization requires stream to be "
                                     "produced by sim.get_file");
        }
        return failure();
      }
    }
  }

  if (livePrintRequests.empty()) {
    for (auto *op : sourceOpsToErase)
      if (op)
        op->erase();
    cleanupDeadPrintArtifacts(cleanupList);
    return success();
  }

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
  for (auto *fragment : allFStringFragments) {
    auto original = fragment->getResult(0);
    if (mapping.lookupOrNull(original))
      continue;
    auto *cloned = builder.clone(*fragment, mapping);
    mapping.map(original, cloned->getResult(0));
  }

  for (auto getFileOp : getFileOps) {
    auto &fileNameFragments = fileNameFragmentMap[getFileOp];
    Value clonedFileName = rematerializeFormatStringFromFragments(
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
  SmallDenseMap<const PrintProceduralizationRequest *, Value, 4>
      procPrintInputMap;
  // Insert after rematerialized fragments/get_file ops so operands dominate.
  builder.setInsertionPointToEnd(trigOp.getBodyBlock());
  for (auto *request : livePrintRequests) {
    auto &printFragments = printFragmentMap[request];
    procPrintInputMap[request] = rematerializeFormatStringFromFragments(
        printFragments, builder, mapping, request->loc);
  }

  Value prevConditionValue;
  Block *prevConditionBlock = nullptr;
  for (auto *request : livePrintRequests) {
    if (alwaysEnabledConditions.contains(request->condition)) {
      prevConditionValue = Value();
      prevConditionBlock = nullptr;
      builder.setInsertionPointToEnd(trigOp.getBodyBlock());
    } else {
      auto condArg = mapping.lookup(request->condition);
      auto *condBlock =
          getOrCreateConditionBlock(builder, trigOp, request->loc, condArg,
                                    prevConditionValue, prevConditionBlock);
      builder.setInsertionPoint(condBlock->getTerminator());
    }

    Value procPrintStream;
    if (auto stream = request->stream) {
      procPrintStream = mapping.lookupOrNull(stream);
      if (!procPrintStream) {
        emitProceduralizationError(*request,
                                   "proceduralization failed to rematerialize "
                                   "stream");
        return failure();
      }
      if (auto *streamDef = stream.getDefiningOp())
        cleanupList.push_back(streamDef);
    }

    PrintFormattedProcOp::create(builder, request->loc,
                                 procPrintInputMap[request], procPrintStream);
    if (auto *inputDef = request->input.getDefiningOp())
      cleanupList.push_back(inputDef);
    if (request->cleanupRoot)
      sourceOpsToErase.push_back(request->cleanupRoot);
  }

  for (auto *op : sourceOpsToErase)
    if (op)
      op->erase();
  cleanupDeadPrintArtifacts(cleanupList);
  return success();
}

void ProceduralizeSimPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");
  printRequestMap.clear();

  auto theModule = getOperation();
  // Collect print operations grouped by their clock, preserving IR order.
  theModule.walk<mlir::WalkOrder::PreOrder>([&](PrintFormattedOp op) {
    printRequestMap[op.getClock()].push_back({op.getLoc(), op.getInput(),
                                              op.getCondition(), op.getStream(),
                                              op, op});
  });

  // Create a hw::TriggeredOp for each clock.
  for (auto &[clock, requests] : printRequestMap) {
    OpBuilder builder(requests.back().anchorOp);
    if (failed(proceduralizePrintsForClock(builder, clock, requests))) {
      signalPassFailure();
      return;
    }
  }
}
