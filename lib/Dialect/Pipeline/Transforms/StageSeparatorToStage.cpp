//===- StageSeparatorToStage.cpp - Explicit regs pass ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the delimiter-to-stage lowering pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Support/BackedgeBuilder.h"

#include "llvm/Support/Debug.h"

#include <memory>

using namespace mlir;
using namespace circt;
using namespace pipeline;

namespace {

class StageSeparatorToStagePass
    : public StageSeparatorToStageBase<StageSeparatorToStagePass> {
public:
  void runOnOperation() override;

private:
  // Returns the operations which resides within each stage-like operation.
  llvm::DenseMap<StageLike, llvm::SmallVector<Operation *>> gatherStageOps();

  // Returns the next stage-like operation wrt. the `fromOp`.
  StageLike getNextStageLike(StageLike fromOp = nullptr);

  // The pipeline operation to be converted.
  PipelineOp pipeline;
};

} // end anonymous namespace

llvm::DenseMap<StageLike, llvm::SmallVector<Operation *>>
StageSeparatorToStagePass::gatherStageOps() {
  llvm::DenseMap<StageLike, llvm::SmallVector<Operation *>> stageOps;
  StageLike currentStageLike = getNextStageLike();
  for (auto &op : pipeline.getBodyBlock()->getOperations()) {
    if (auto stageLike = dyn_cast<StageLike>(&op)) {
      assert(currentStageLike == stageLike &&
             "Expected to encounter the current stage like operation");
      currentStageLike = getNextStageLike(stageLike);
    } else
      stageOps[currentStageLike].push_back(&op);
  };
  return stageOps;
}

StageLike StageSeparatorToStagePass::getNextStageLike(StageLike fromOp) {
  auto stageLikes = pipeline.getBodyBlock()->getOps<StageLike>();
  if (!fromOp)
    return *stageLikes.begin();

  auto it = llvm::find(stageLikes, fromOp);
  assert(it != stageLikes.end() && "stage not found");
  if (std::next(it) == stageLikes.end())
    return nullptr;
  return *std::next(it);
}

void StageSeparatorToStagePass::runOnOperation() {
  pipeline = getOperation();
  OpBuilder b(getOperation().getContext());
  StageLike nextStageLike = nullptr;

  auto stageOps = gatherStageOps();
  nextStageLike = getNextStageLike();

  // Maintain the set of constant operations in the pipeline - these
  // eventually will need to be sunk into the stages which reference them.
  llvm::SmallVector<Operation *> constants;

  ValueRange currentStageInputs = pipeline.getInnerInputs();
  Value currentStageEnable = pipeline.getInnerEnable();

  // 1. iterate over stage delimiter operations
  // 2. build stages
  // 3. move operations in between stages into the new stage ops.
  for (auto stageLikeOp :
       llvm::make_early_inc_range(pipeline.getOps<StageSeparatingRegOp>())) {
    b.setInsertionPoint(stageLikeOp);
    StageOp currentStage =
        b.create<StageOp>(stageLikeOp->getLoc(), currentStageEnable,
                          currentStageInputs, stageLikeOp.getInputTypes());

    // Replace the stage-like op with the new stage op.
    bool isReturn = isa<ReturnOp>(stageLikeOp);
    if (!isReturn) {
      stageLikeOp->replaceAllUsesWith(currentStage);
      stageLikeOp.erase();
    }

    // Move non-constant ops from the stageOps map into the StageOp wherein
    // they will reside.
    auto stageReturn =
        cast<StageReturnOp>(currentStage.getBodyBlock()->getTerminator());
    for (auto *op : stageOps[stageLikeOp]) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        constants.push_back(op);
      else
        op->moveBefore(stageReturn);
    }

    // Finalize the current stage by adjusting the stage return value.
    stageReturn.setOperands(currentStage.getInnerEnable(),
                            stageLikeOp.getInputs(),
                            /*passthroughs=*/{});

    // Replace usages of the stage inputs inside the stage with the stage
    // inner inputs.
    for (auto [stageInput, innerInput] :
         llvm::zip(currentStageInputs, currentStage.getInnerInputs()))
      stageInput.replaceAllUsesExcept(innerInput, currentStage);

    // Update the outputs fed to the next stage.
    currentStageInputs = currentStage.getOutputs();
    currentStageEnable = currentStage.getValid();
  }

  // Constant sinking - copy constant ops into each stage that references
  // them.
  for (auto *op : constants) {
    llvm::DenseMap<StageOp, llvm::SmallVector<OpOperand *>> stageUsers;
    for (auto &use : op->getUses()) {
      auto parentStage = dyn_cast<StageOp>(use.getOwner()->getParentOp());
      if (!parentStage)
        continue;

      stageUsers[parentStage].push_back(&use);
    }

    for (auto [stage, uses] : stageUsers) {
      auto *copiedConstant = op->clone();
      copiedConstant->moveBefore(&stage.getBodyBlock()->front());
      for (auto *use : uses)
        use->set(copiedConstant->getResult(0));
    }
  }
}

std::unique_ptr<mlir::Pass> circt::pipeline::createStageSeparatorToStagePass() {
  return std::make_unique<StageSeparatorToStagePass>();
}
