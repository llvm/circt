//===- ConvertSimTriggeredToHW.cpp - Lower sim.triggered to hw.triggered --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert top-level sim.triggered ops in a hw.module body to hw.triggered.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "circt/Support/SparseOpSCC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_CONVERTSIMTRIGGEREDTOHW
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace llvm;
using namespace circt;
using namespace sim;

namespace {

static LogicalResult collectFormatStringFragments(
    Value formatString, Operation *anchorOp,
    SmallVectorImpl<Operation *> &fragmentList,
    SmallSetVector<Operation *, 8> &allFStringFragments,
    llvm::SetVector<Value> &captures) {
  SmallVector<Value> flatString;
  if (auto concatInput = formatString.getDefiningOp<FormatStringConcatOp>()) {
    if (failed(concatInput.getFlattenedInputs(flatString))) {
      anchorOp->emitError("cyclic sim.fmt.concat is unsupported");
      return failure();
    }
  } else {
    flatString.push_back(formatString);
  }

  for (auto fragment : flatString) {
    auto *fmtOp = fragment.getDefiningOp();
    if (!fmtOp) {
      anchorOp->emitError("block argument format strings are unsupported");
      return failure();
    }
    fragmentList.push_back(fmtOp);
    allFStringFragments.insert(fmtOp);
    if (auto fmtVal = getFormattedValue(fmtOp))
      captures.insert(fmtVal);
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
    assert(cloned && "missing cloned format fragment");
    operands.push_back(cloned);
  }
  if (operands.size() == 1)
    return operands.front();
  return builder.createOrFold<FormatStringConcatOp>(loc, operands);
}

static void cloneBodyOperations(Block *source, Block *dest, OpBuilder &builder,
                                IRMapping &mapping) {
  for (Operation &op : *source) {
    if (dest->mightHaveTerminator()) {
      if (auto *terminator = dest->getTerminator())
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(dest);
    } else {
      builder.setInsertionPointToEnd(dest);
    }
    auto *cloned = builder.clone(op, mapping);
    for (auto [original, result] :
         llvm::zip(op.getResults(), cloned->getResults()))
      mapping.map(original, result);
  }
}

static void cleanupDeadSimFmtOps(ArrayRef<Operation *> seedOps) {
  auto filter = [](Operation *op, OpOperand &operand) {
    return isa<FormatStringType>(operand.get().getType()) &&
           isa_and_present<SimDialect>(op->getDialect());
  };

  SparseOpSCC<OpSCCDirection::Backward> sccs(filter);
  sccs.visit(seedOps);
  assert(sccs.getNumCyclicSCCs() == 0 &&
         "Cyclic graph should have been rejected");

  for (OpSCC entry : sccs.reverseTopological()) {
    auto *op = cast<Operation *>(entry);
    if (op->use_empty())
      op->erase();
    else
      op->emitWarning("sim format op still has users after conversion");
  }
}

struct ConvertSimTriggeredToHWPass
    : impl::ConvertSimTriggeredToHWBase<ConvertSimTriggeredToHWPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult convertTriggered(TriggeredOp op);

  SmallVector<Operation *> cleanupSeeds;
};

} // namespace

LogicalResult ConvertSimTriggeredToHWPass::convertTriggered(TriggeredOp op) {
  llvm::SetVector<Value> captures;
  mlir::getUsedValuesDefinedAbove(op.getBody(), op.getBody(), captures);
  if (auto condition = op.getCondition())
    captures.insert(condition);

  SmallDenseMap<Value, SmallVector<Operation *, 4>, 4> formatStringFragmentMap;
  SmallSetVector<Operation *, 8> allFStringFragments;
  SmallVector<Value> externalFormatStrings;
  for (Value capture : captures)
    if (isa<FormatStringType>(capture.getType()))
      externalFormatStrings.push_back(capture);

  for (Value formatString : externalFormatStrings) {
    captures.remove(formatString);
    if (auto *defOp = formatString.getDefiningOp())
      cleanupSeeds.push_back(defOp);
    auto &fragments = formatStringFragmentMap[formatString];
    if (failed(collectFormatStringFragments(formatString, op, fragments,
                                            allFStringFragments, captures)))
      return failure();
  }

  SmallVector<Value> capturedVec(captures.begin(), captures.end());

  OpBuilder builder(op);
  auto event = hw::EventControlAttr::get(builder.getContext(),
                                         hw::EventControl::AtPosEdge);
  auto trigger =
      builder.createOrFold<seq::FromClockOp>(op.getLoc(), op.getClock());
  auto converted = hw::TriggeredOp::create(builder, op.getLoc(), event, trigger,
                                           capturedVec);

  IRMapping mapping;
  for (auto [captured, arg] :
       llvm::zip(capturedVec, converted.getInnerInputs()))
    mapping.map(captured, arg);

  Block *destBlock = converted.getBodyBlock();
  builder.setInsertionPointToStart(destBlock);
  for (auto *fragment : allFStringFragments) {
    auto original = fragment->getResult(0);
    if (mapping.lookupOrNull(original))
      continue;
    auto *cloned = builder.clone(*fragment, mapping);
    mapping.map(original, cloned->getResult(0));
  }
  for (auto &[formatString, fragments] : formatStringFragmentMap)
    mapping.map(formatString, rematerializeFormatStringFromFragments(
                                  fragments, builder, mapping,
                                  formatString.getLoc()));

  if (auto condition = op.getCondition()) {
    builder.setInsertionPointToEnd(destBlock);
    auto outerIf =
        mlir::scf::IfOp::create(builder, op.getLoc(), TypeRange{},
                                mapping.lookup(condition), true, false);
    builder.setInsertionPointToStart(&outerIf.getThenRegion().front());
    mlir::scf::YieldOp::create(builder, op.getLoc());
    destBlock = &outerIf.getThenRegion().front();
  }

  cloneBodyOperations(op.getBodyBlock(), destBlock, builder, mapping);
  op.erase();
  return success();
}

void ConvertSimTriggeredToHWPass::runOnOperation() {
  cleanupSeeds.clear();
  hw::HWModuleOp module = getOperation();
  SmallVector<TriggeredOp> triggeredOps;
  for (Operation &op : *module.getBodyBlock())
    if (auto triggered = dyn_cast<TriggeredOp>(op))
      triggeredOps.push_back(triggered);

  if (triggeredOps.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  for (auto triggered : triggeredOps)
    if (failed(convertTriggered(triggered))) {
      signalPassFailure();
      return;
    }

  cleanupDeadSimFmtOps(cleanupSeeds);
}
