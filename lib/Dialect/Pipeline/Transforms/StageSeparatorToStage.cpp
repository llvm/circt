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

using namespace mlir;
using namespace circt;
using namespace pipeline;

namespace {

class StageSeparatorToStagePass
    : public StageSeparatorToStageBase<StageSeparatorToStagePass> {
public:
  void runOnOperation() override;

private:
  // Returns the operations which resides within each stage.
  llvm::DenseMap<StageSeparatingRegOp, llvm::SmallVector<Operation *>>
  gatherStageOps();

  // The pipeline operation to be converted.
  PipelineOp pipeline;
};

} // end anonymous namespace

llvm::DenseMap<StageSeparatingRegOp, llvm::SmallVector<Operation *>>
StageSeparatorToStagePass::gatherStageOps() {
  auto stageSeparators = pipeline.getOps<StageSeparatingRegOp>();
  StageSeparatingRegOp currentSeparator = *stageSeparators.begin();
  auto setNextSeparator = [&]() {
    auto next =
        std::next(llvm::find_if(stageSeparators, [&](StageSeparatingRegOp op) {
          return op == currentSeparator;
        }));
    if (next == stageSeparators.end()) {
      currentSeparator = nullptr;
      return;
    }
    currentSeparator = *next;
  };

  llvm::DenseMap<StageSeparatingRegOp, llvm::SmallVector<Operation *>> stageOps;
  for (auto &op : pipeline.getBodyBlock()->getOperations()) {
    if (!currentSeparator) {
      // End of pipeline stages - 'op' is either a return value or
      // operations residing in the output stage of the pipeline, which are
      // not to be registered/placed in an explicit stage.
      break;
    }
    if (auto separator = dyn_cast<StageSeparatingRegOp>(&op)) {
      assert(currentSeparator == separator &&
             "Expected to encounter the current stage like operation");
      setNextSeparator();
    } else {
      stageOps[currentSeparator].push_back(&op);
    }
  };
  return stageOps;
}

void StageSeparatorToStagePass::runOnOperation() {
  pipeline = getOperation();
  OpBuilder b(getOperation().getContext());

  auto stageOps = gatherStageOps();

  // Maintain the set of constant operations in the pipeline - these
  // eventually will need to be sunk into the stages which reference them.
  llvm::SmallVector<Operation *> constants;

  ValueRange currentStageInputs = pipeline.getInnerInputs();
  Value currentStageEnable = pipeline.getInnerEnable();

  // 1. iterate over stage delimiter operations
  // 2. build stages
  // 3. move operations in between stages into the new stage ops.
  for (auto stageSep :
       llvm::make_early_inc_range(pipeline.getOps<StageSeparatingRegOp>())) {
    b.setInsertionPoint(stageSep);
    StageOp currentStage =
        b.create<StageOp>(stageSep->getLoc(), currentStageEnable,
                          currentStageInputs, stageSep.getInputs().getTypes());

    // Replace the stage separator with the new stage op.
    stageSep->replaceAllUsesWith(currentStage);
    stageSep.erase();

    // Move non-constant ops from the stageOps map into the StageOp wherein
    // they will reside.
    auto stageReturn =
        cast<StageReturnOp>(currentStage.getBodyBlock()->getTerminator());
    for (auto *op : stageOps[stageSep]) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        constants.push_back(op);
      else
        op->moveBefore(stageReturn);
    }

    // Finalize the current stage by adjusting the stage return value.
    stageReturn.setOperands(currentStage.getInnerEnable(), stageSep.getInputs(),
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
      b.setInsertionPointToStart(stage.getBodyBlock());
      b.insert(copiedConstant);
      for (auto *use : uses)
        use->set(copiedConstant->getResult(0));
    }
  }
}

std::unique_ptr<mlir::Pass> circt::pipeline::createStageSeparatorToStagePass() {
  return std::make_unique<StageSeparatorToStagePass>();
}
