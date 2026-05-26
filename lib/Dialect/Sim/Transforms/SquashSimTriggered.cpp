//===- SquashSimTriggered.cpp - Merge sim.triggered ops by clock ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Merge multiple sim.triggered ops in the same block that share a clock into a
// single sim.triggered. Conditions that cannot be represented on the merged op
// directly are materialized as scf.if operations inside the merged body.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_SQUASHSIMTRIGGERED
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace llvm;
using namespace circt;
using namespace sim;

namespace {

static Block *getOrCreateConditionBlock(OpBuilder &builder, Location loc,
                                        Value condition,
                                        Value &prevConditionValue,
                                        Block *&prevConditionBlock) {
  if (condition != prevConditionValue)
    prevConditionBlock = nullptr;

  if (prevConditionBlock)
    return prevConditionBlock;

  auto ifOp = mlir::scf::IfOp::create(builder, loc, TypeRange{}, condition,
                                      true, false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::scf::YieldOp::create(builder, loc);
  prevConditionValue = condition;
  prevConditionBlock = builder.getBlock();
  return prevConditionBlock;
}

static void moveBodyOperations(Block *source, Block *dest,
                               Block::iterator insertPt) {
  dest->getOperations().splice(insertPt, source->getOperations(),
                               source->begin(), source->end());
}

static hw::TriggeredOp convertTriggeredToHW(TriggeredOp op) {
  mlir::IRRewriter builder(op->getContext());
  builder.setInsertionPoint(op);
  auto captures = mlir::makeRegionIsolatedFromAbove(builder, op.getBody());

  Value conditionArg;
  if (auto condition = op.getCondition()) {
    auto *it = llvm::find(captures, condition);
    if (it == captures.end()) {
      conditionArg = condition;
      op.getBodyBlock()->addArgument(condition.getType(), condition.getLoc());
      captures.push_back(condition);
    } else {
      conditionArg = *it;
    }
  }

  unsigned conditionArgIndex = 0;
  if (conditionArg)
    conditionArgIndex = llvm::find(captures, conditionArg) - captures.begin();

  auto event = hw::EventControlAttr::get(builder.getContext(),
                                         hw::EventControl::AtPosEdge);
  auto trigger =
      builder.createOrFold<seq::FromClockOp>(op.getLoc(), op.getClock());
  auto converted =
      hw::TriggeredOp::create(builder, op.getLoc(), event, trigger, captures);

  converted.getBody().takeBody(op.getBody());
  if (op.getCondition()) {
    auto arg = converted.getBodyBlock()->getArgument(conditionArgIndex);
    builder.setInsertionPointToStart(converted.getBodyBlock());
    auto ifOp = mlir::scf::IfOp::create(builder, op.getLoc(), TypeRange{}, arg,
                                        true, false);
    auto *thenBlock = &ifOp.getThenRegion().front();
    builder.setInsertionPointToEnd(thenBlock);
    mlir::scf::YieldOp::create(builder, op.getLoc());
    thenBlock->getOperations().splice(
        Block::iterator(thenBlock->getTerminator()),
        converted.getBodyBlock()->getOperations(),
        std::next(Block::iterator(ifOp.getOperation())),
        converted.getBodyBlock()->end());
  }

  op.erase();
  return converted;
}

struct SquashSimTriggeredPass
    : impl::SquashSimTriggeredBase<SquashSimTriggeredPass> {
public:
  using SquashSimTriggeredBase::SquashSimTriggeredBase;
  void runOnOperation() override;

private:
  bool squashTriggeredOpsInBlock(Block &block);
  void convertTriggeredOpsInBlock(Block &block);
};

} // namespace

bool SquashSimTriggeredPass::squashTriggeredOpsInBlock(Block &block) {
  // Group all top-level sim.triggered ops in this block by their clock.
  SmallMapVector<Value, SmallVector<TriggeredOp>, 2> triggerMap;
  for (Operation &op : block)
    if (auto triggered = dyn_cast<TriggeredOp>(op))
      triggerMap[triggered.getClock()].push_back(triggered);

  bool changed = false;
  for (auto &[clock, triggers] : triggerMap) {
    if (triggers.size() < 2)
      continue;

    SmallVector<Location> locs;
    locs.reserve(triggers.size());

    bool hasUnconditionalTrigger = false;
    Value commonCondition;
    bool allConditionsIdentical = true;
    for (auto triggered : triggers) {
      locs.push_back(triggered.getLoc());
      if (auto condition = triggered.getCondition()) {
        if (!commonCondition)
          commonCondition = condition;
        else if (condition != commonCondition)
          allConditionsIdentical = false;
      } else {
        hasUnconditionalTrigger = true;
      }
    }

    // Hoist a shared condition to the merged sim.triggered when possible.
    Value outerCondition;
    if (!hasUnconditionalTrigger && allConditionsIdentical)
      outerCondition = commonCondition;

    OpBuilder builder(triggers.back());
    auto fusedLoc = builder.getFusedLoc(locs);
    auto mergedTriggered =
        TriggeredOp::create(builder, fusedLoc, clock, outerCondition);

    Value prevConditionValue;
    Block *prevConditionBlock = nullptr;
    for (auto triggered : triggers) {
      auto condition = triggered.getCondition();
      auto *sourceBlock = triggered.getBodyBlock();

      // Conditions not represented on the outer op become inner scf.if guards.
      if (!sourceBlock->empty() && condition && condition != outerCondition) {
        builder.setInsertionPointToEnd(mergedTriggered.getBodyBlock());
        auto *condBlock =
            getOrCreateConditionBlock(builder, triggered.getLoc(), condition,
                                      prevConditionValue, prevConditionBlock);
        moveBodyOperations(sourceBlock, condBlock,
                           Block::iterator(condBlock->getTerminator()));
      } else {
        prevConditionValue = Value();
        prevConditionBlock = nullptr;
        moveBodyOperations(sourceBlock, mergedTriggered.getBodyBlock(),
                           mergedTriggered.getBodyBlock()->end());
      }

      triggered.erase();
    }
    changed = true;
  }

  return changed;
}

void SquashSimTriggeredPass::convertTriggeredOpsInBlock(Block &block) {
  SmallVector<TriggeredOp> triggeredOps;
  for (Operation &op : block)
    if (auto triggered = dyn_cast<TriggeredOp>(op))
      triggeredOps.push_back(triggered);

  for (auto triggered : triggeredOps)
    convertTriggeredToHW(triggered);
}

void SquashSimTriggeredPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();
  bool changed = squashTriggeredOpsInBlock(*module.getBodyBlock());
  if (convertToHW) {
    convertTriggeredOpsInBlock(*module.getBodyBlock());
    changed = true;
  }
  if (!changed)
    markAllAnalysesPreserved();
}
