//===- ExplicitRegs.cpp - Explicit regs pass --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the explicit regs pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Support/BackedgeBuilder.h"

using namespace mlir;
using namespace circt;
using namespace pipeline;

namespace {

class ExplicitRegsPass : public ExplicitRegsBase<ExplicitRegsPass> {
public:
  void runOnOperation() override;

private:
  // Recursively routes value v backwards through the pipeline, adding new
  // registers to 'stage' if the value was not already registered in the stage.
  // Returns the registerred version of 'v' through 'stage'.
  Value routeThroughStage(OpOperand &v, Block *stage);

  // A mapping storing whether a given stage register constains a registerred
  // version of a given value. The registered version will be a backedge during
  // pipeline body analysis. Once the entire body has been analyzed, the
  // pipeline.stage operations will be replaced with pipeline.ss.reg
  // operations containing the requested regs, and the backedge will be
  // replaced. MapVector ensures deterministic iteration order, which in turn
  // ensures determinism during stage op IR emission.
  DenseMap<Block *, llvm::MapVector<Value, Backedge>> stageRegMap;

  // A linked list of stages in the pipeline. Allows for easily looking up the
  // predecessor stage of a given stage.
  DenseMap<Block *, Block *> stagePredecessor;

  std::shared_ptr<BackedgeBuilder> bb;
};

} // end anonymous namespace

// NOLINTNEXTLINE(misc-no-recursion)
Value ExplicitRegsPass::routeThroughStage(OpOperand &v, Block *stage) {
  Value retVal = v.get();

  auto regIt = stageRegMap[stage].find(retVal);
  if (regIt != stageRegMap[stage].end()) {
    // 'v' is already registered in 'stage'.
    return regIt->second;
  }

  auto *definingOp = retVal.getDefiningOp();

  // Is the value defined by an op in the current stage?
  if (definingOp && (definingOp->getBlock() == stage))
    return retVal;

  // Is the value a block argument of the current stage?
  if (llvm::is_contained(stage->getArguments(), retVal))
    return retVal;

  // Is the value a constant? If so, we allow it; constants are special cases
  // which are allowed to be used in any stage.
  if (definingOp && definingOp->hasTrait<OpTrait::ConstantLike>())
    return retVal;

  // Value is defined somewhere before the provided stage - route it through the
  // stage, and recurse to the predecessor stage.
  auto regBackedge = bb->get(retVal.getType());
  stageRegMap[stage].insert({retVal, regBackedge});
  retVal = regBackedge;

  // Recurse - recursion will only create a new backedge if necessary.
  auto predStageIt = stagePredecessor.find(stage);
  assert(predStageIt != stagePredecessor.end() &&
         "stage should have been registered before calling this function");
  Block *predecessorStage = predStageIt->second;
  if (predecessorStage)
    routeThroughStage(v, predecessorStage);

  return retVal;
}

void ExplicitRegsPass::runOnOperation() {
  ScheduledPipelineOp pipeline = getOperation();
  OpBuilder b(getOperation().getContext());
  bb = std::make_shared<BackedgeBuilder>(b, getOperation().getLoc());

  // A list of stages in the pipeline in the order which they appear.
  SmallVector<Block *> stageList;
  Block *currStage = nullptr;
  // Iterate over the pipeline body in-order (!).
  for (Block *stage : pipeline.getOrderedStages()) {
    stagePredecessor[stage] = currStage;
    currStage = stage;

    for (auto &op : *stage) {
      // Check the operands of this operation to see if any of them cross a
      // stage boundary.
      for (OpOperand &operand : op.getOpOperands()) {
        Value reroutedValue = routeThroughStage(operand, currStage);
        if (reroutedValue != operand.get())
          op.setOperand(operand.getOperandNumber(), reroutedValue);
      }
    }
  }

  auto *ctx = &getContext();

  // All values have been recorded through the stages. Now, add registers to the
  // stage blocks.
  for (auto &[stage, regMap] : stageRegMap) {
    Block *predStage = stagePredecessor[stage];

    // Gather register inputs to this stage, either from a predecessor stage
    // or from the original op.
    llvm::SmallVector<Value> regIns;
    for (auto &[value, backedge] : regMap) {
      if (predStage) {
        // Grab the value if registerred through the predecessor op, else,
        // use the raw value.
        auto predRegIt = stageRegMap[predStage].find(value);
        if (predRegIt != stageRegMap[predStage].end()) {
          regIns.push_back(predRegIt->second);
          continue;
        }
      }
      // Not in predecessor stage - must be the original value.
      regIns.push_back(value);
    }

    // ... add arguments to the stage
    stage->addArguments(
        ValueRange(regIns).getTypes(),
        llvm::SmallVector<Location>(regIns.size(), UnknownLoc::get(ctx)));

    // Replace backedges with the stage arguments.
    for (auto it : llvm::enumerate(regMap)) {
      auto index = it.index();
      auto &[value, backedge] = it.value();
      backedge.setValue(stage->getArgument(index));
    }

    // And append arguments to the predecessor stage terminator, if necessary.
    if (predStage) {
      StageOp predTerminator = cast<StageOp>(predStage->getTerminator());
      predTerminator.getRegistersMutable().append(regIns);
    }
  }

  // Clear internal state. See https://github.com/llvm/circt/issues/3235
  stageRegMap.clear();
  stagePredecessor.clear();
}

std::unique_ptr<mlir::Pass> circt::pipeline::createExplicitRegsPass() {
  return std::make_unique<ExplicitRegsPass>();
}
