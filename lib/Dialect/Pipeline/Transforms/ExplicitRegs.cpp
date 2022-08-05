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
  Value routeThroughStage(OpOperand &v, PipelineStageOp stage);

  // A mapping storing whether a given stage register constains a registerred
  // version of a given value. The registered version will be a backedge during
  // pipeline body analysis. Once the entire body has been analyzed, the
  // pipeline.stage operations will be replaced with pipeline.stage.register
  // operations containing the requested regs, and the backedge will be
  // replaced. MapVector ensures deterministic iteration order, which in turn
  // ensures determinism during stage op IR emission.
  DenseMap<PipelineStageOp, llvm::MapVector<Value, Backedge>> stageRegMap;

  // A linked list of stages in the pipeline. Allows for easily looking up the
  // predecessor stage of a given stage.
  DenseMap<PipelineStageOp, PipelineStageOp> stagePredecessor;

  std::shared_ptr<BackedgeBuilder> bb;
};

} // end anonymous namespace

// NOLINTNEXTLINE(misc-no-recursion)
Value ExplicitRegsPass::routeThroughStage(OpOperand &v, PipelineStageOp stage) {
  if (!stage) {
    // Recursive base case - nothing to route (v is a block operand).
    return v.get();
  }

  auto regIt = stageRegMap[stage].find(v.get());
  if (regIt != stageRegMap[stage].end()) {
    // 'v' is already registered in 'stage'.
    return regIt->second;
  }

  auto *definingOp = v.get().getDefiningOp();
  auto blockPos = [&](Operation *op) {
    return std::distance(op->getBlock()->begin(), op->getIterator());
  };

  // If no defining op (a block operand), define the defining position as a
  // negative value (indicating "earliest possible position").
  int pDef = definingOp ? blockPos(definingOp) : -1;
  int pStage = blockPos(stage);

  Value retVal = v.get();
  if (pDef < pStage) {
    // Value is defined before the provided stage - route it through the stage.
    auto regBackedge = bb->get(v.get().getType());
    stageRegMap[stage].insert({v.get(), regBackedge});
    retVal = regBackedge;
  } else {
    // Value is defined after the provided stage - early exit here.
    return retVal;
  }

  // Recurse - recursion will only create a new backedge if necessary.
  auto predStageIt = stagePredecessor.find(stage);
  assert(predStageIt != stagePredecessor.end() &&
         "stage should have been registered before calling this function");
  PipelineStageOp predecessorStage = predStageIt->second;
  if (predecessorStage)
    routeThroughStage(v, predecessorStage);

  return retVal;
}

void ExplicitRegsPass::runOnOperation() {
  auto pipeline = getOperation();
  OpBuilder b(getOperation().getContext());
  bb = std::make_shared<BackedgeBuilder>(b, getOperation().getLoc());

  // A list of stages in the pipeline in the order which they appear.
  SmallVector<PipelineStageOp> stageList;
  PipelineStageOp currStage = nullptr;
  // Iterate over the pipeline body in-order (!).
  for (auto &op : *pipeline.getBody()) {
    if (auto stageOp = dyn_cast<PipelineStageOp>(&op)) {
      stagePredecessor[stageOp] = currStage;
      currStage = stageOp;
      continue;
    }

    // Check the operands of this operation to see if any of them cross a
    // stage boundary.
    for (OpOperand &operand : op.getOpOperands()) {
      Value reroutedValue = routeThroughStage(operand, currStage);
      if (reroutedValue != operand.get())
        op.setOperand(operand.getOperandNumber(), reroutedValue);
    }
  }

  // All values have been recorded through the stages. Now, replace the
  // stages with new stage operations containing the required registers.
  for (auto &[stageOp, regMap] : stageRegMap) {
    b.setInsertionPoint(stageOp);
    auto predStageOp = stagePredecessor[stageOp];

    // Gather register inputs to this stage, either from a predecessor stage
    // or from the original op.
    llvm::SmallVector<Value> regIns;
    for (auto &[value, backedge] : regMap) {
      if (predStageOp) {
        // Grab the value if registerred through the predecessor op, else,
        // use the raw value.
        auto predRegIt = stageRegMap[predStageOp].find(value);
        if (predRegIt != stageRegMap[predStageOp].end()) {
          regIns.push_back(predRegIt->second);
          continue;
        }
      }
      // Not in predecessor stage - must be the original value.
      regIns.push_back(value);
    }

    auto newStageOp = b.create<PipelineStageRegisterOp>(stageOp.getLoc(),
                                                        stageOp.when(), regIns);
    stageOp.valid().replaceAllUsesWith(newStageOp.valid());

    // Replace backedges with the outputs of the new stage.
    for (auto &it : llvm::enumerate(regMap)) {
      auto index = it.index();
      auto &[value, backedge] = it.value();
      backedge.setValue(newStageOp.regOuts()[index]);
    }
    stageOp.erase();
  }

  // Clear internal state. See https://github.com/llvm/circt/issues/3235
  stageRegMap.clear();
  stagePredecessor.clear();
}

std::unique_ptr<mlir::Pass> circt::pipeline::createExplicitRegsPass() {
  return std::make_unique<ExplicitRegsPass>();
}
