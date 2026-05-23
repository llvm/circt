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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
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

struct ConvertSimTriggeredToHWPass
    : impl::ConvertSimTriggeredToHWBase<ConvertSimTriggeredToHWPass> {
public:
  void runOnOperation() override;

private:
  void convertTriggered(TriggeredOp op);
};

} // namespace

void ConvertSimTriggeredToHWPass::convertTriggered(TriggeredOp op) {
  llvm::SetVector<Value> captures;
  mlir::getUsedValuesDefinedAbove(op.getBody(), op.getBody(), captures);
  if (auto condition = op.getCondition())
    captures.insert(condition);
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
}

void ConvertSimTriggeredToHWPass::runOnOperation() {
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
    convertTriggered(triggered);
}
