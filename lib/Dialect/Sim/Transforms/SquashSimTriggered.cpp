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
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
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

struct SquashSimTriggeredPass
    : impl::SquashSimTriggeredBase<SquashSimTriggeredPass> {
public:
  void runOnOperation() override;

private:
  bool squashTriggeredOpsInBlock(Block &block);
};

} // namespace

bool SquashSimTriggeredPass::squashTriggeredOpsInBlock(Block &block) {
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

void SquashSimTriggeredPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();
  if (!squashTriggeredOpsInBlock(*module.getBodyBlock()))
    markAllAnalysesPreserved();
}
