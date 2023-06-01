//===- PipelineToHW.cpp - Translate Pipeline into HW ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Pipeline to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/PipelineToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Pipeline/Pipeline.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace pipeline;

// A class for generalizing pipeline lowering for both the inline and outlined
// implementation.
class PipelineLowering {
public:
  PipelineLowering(size_t pipelineID, PipelineOp pipeline, OpBuilder &builder)
      : pipelineID(pipelineID), pipeline(pipeline), builder(builder) {
    parentClk = pipeline.getClock();
    parentRst = pipeline.getReset();
    parentModule = pipeline->getParentOfType<hw::HWModuleOp>();
  }
  virtual ~PipelineLowering() = default;

  virtual LogicalResult run() = 0;

  LogicalResult runInner() {
    for (auto &op : llvm::make_early_inc_range(*pipeline.getBodyBlock())) {
      llvm::TypeSwitch<Operation *, void>(&op)
          .Case<StageOp>([&](auto stage) { lowerStage(stage); })
          .Case<pipeline::ReturnOp>(
              [&](auto ret) { pipeline->replaceAllUsesWith(ret.getInputs()); })
          .Default([&](auto op) {
            // Free-standing operations are moved to the current insertion
            // point. It is the responsability of the implementation to maintain
            // the insertion point.
            op->moveBefore(builder.getInsertionBlock(),
                           builder.getInsertionPoint());
          });
    }

    return success();
  }

  struct StageReturns {
    llvm::SmallVector<Value> regs;
    llvm::SmallVector<Value> passthroughs;
    Value valid;
  };

  virtual void lowerStage(StageOp stage) = 0;
  StageReturns emitStageBody(StageOp stage, Value clock, Value reset) {
    auto stageRetOp =
        cast<StageReturnOp>(stage.getBodyBlock()->getTerminator());

    // Move the stage operations into the current insertion point.
    for (auto &op : llvm::make_early_inc_range(*stage.getBodyBlock())) {
      if (&op == stageRetOp)
        continue;
      op.moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
    }

    StageReturns rets;

    // Build data registers
    auto stageRegPrefix = getStageRegPrefix(stage.index());
    for (auto it : llvm::enumerate(stageRetOp.getRegs())) {
      auto regIdx = it.index();
      auto regIn = it.value();
      auto regName = builder.getStringAttr(stageRegPrefix.strref() + "_reg" +
                                           std::to_string(regIdx));
      auto reg = builder.create<seq::CompRegOp>(
          stageRetOp->getLoc(), regIn.getType(), regIn, clock, regName, reset,
          Value(), StringAttr());
      rets.regs.push_back(reg);
    }

    // Build valid register
    auto validRegName =
        builder.getStringAttr(stageRegPrefix.strref() + "_valid");
    auto validReg = builder.create<seq::CompRegOp>(
        stageRetOp->getLoc(), builder.getI1Type(), stageRetOp.getValid(), clock,
        validRegName, reset, Value(), StringAttr());
    rets.valid = validReg;
    return rets;
  }

  // Returns a string to be used as a prefix for all stage registers.
  virtual StringAttr getStageRegPrefix(size_t stageIdx) = 0;

protected:
  // Parent module clock.
  Value parentClk;
  // Parent module reset.
  Value parentRst;
  // ID of the current pipeline, used for naming.
  size_t pipelineID;
  // The current pipeline to be converted.
  PipelineOp pipeline;

  // The module wherein the pipeline resides.
  hw::HWModuleOp parentModule;

  OpBuilder &builder;

  // Name of this pipeline - used for naming stages and registers.
  // Implementation defined.
  std::string pipelineName;
};

class PipelineInlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;

  StringAttr getStageRegPrefix(size_t stageIdx) override {
    return builder.getStringAttr(pipelineName + "_s" +
                                 std::to_string(stageIdx));
  }

  LogicalResult run() override {
    pipelineName = "p" + std::to_string(pipelineID);

    // Replace uses of the pipeline internal inputs with the pipeline inputs.
    for (auto [outer, inner] :
         llvm::zip(pipeline.getInputs(), pipeline.getInnerInputs()))
      inner.replaceAllUsesWith(outer);

    // All operations should go directly before the pipeline op, into the
    // parent module.
    builder.setInsertionPoint(pipeline);
    if (failed(runInner()))
      return failure();

    // Replace the pipeline with the arguments of the pipeline return op.
    auto returnOp = cast<pipeline::ReturnOp>(pipeline.getBodyBlock()->back());
    pipeline.replaceAllUsesWith(returnOp.getInputs());
    pipeline.erase();
    return success();
  }

  void lowerStage(StageOp stage) override {
    OpBuilder::InsertionGuard guard(builder);
    // Replace the internal stage inputs with the external stage inputs.
    for (auto [vInput, vBarg] :
         llvm::zip(stage.getInnerInputs(), stage.getInputs()))
      vInput.replaceAllUsesWith(vBarg);

    // Replace the internal stage enable with the external stage enable.
    stage.getInnerEnable().replaceAllUsesWith(stage.getEnable());

    // Move stage operations into the current module.
    builder.setInsertionPoint(pipeline);
    auto stageRets = emitStageBody(stage, parentClk, parentRst);

    for (auto [vOutput, vResult] :
         llvm::zip(stage.getOutputs(), stageRets.regs))
      vOutput.replaceAllUsesWith(vResult);

    // and the valid...
    stage.getValid().replaceAllUsesWith(stageRets.valid);
  }
};

class PipelineOutlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;

  StringAttr getStageRegPrefix(size_t stageIdx) override {
    return builder.getStringAttr("s" + std::to_string(stageIdx));
  }

  LogicalResult run() override {
    pipelineName =
        (parentModule.getName() + "_p" + std::to_string(pipelineID)).str();

    // Build the top-level pipeline module.
    pipelineMod = buildPipelineLike(
        pipelineName, pipeline.getInputs().getTypes(),
        pipeline.getResults().getTypes(), /*withEnableValid=*/false);
    pipelineClk = pipelineMod.getBody().front().getArgument(
        pipelineMod.getBody().front().getNumArguments() - 2);
    pipelineRst = pipelineMod.getBody().front().getArgument(
        pipelineMod.getBody().front().getNumArguments() - 1);

    // Instantiate the pipeline in the parent module.
    builder.setInsertionPoint(pipeline);
    llvm::SmallVector<Value, 4> pipelineOperands;
    llvm::append_range(pipelineOperands, pipeline.getOperands());
    auto pipelineInst = builder.create<hw::InstanceOp>(
        pipeline.getLoc(), pipelineMod,
        builder.getStringAttr(pipelineMod.getName()), pipelineOperands);

    // Replace the top-level pipeline results with the pipeline instance
    // results.
    pipeline.replaceAllUsesWith(pipelineInst.getResults());

    // Remap the internal inputs of the pipeline to the pipeline module block
    // arguments.
    for (auto [vInput, vBarg] :
         llvm::zip(pipeline.getInnerInputs(),
                   pipelineMod.getBody().front().getArguments().drop_back(2)))
      vInput.replaceAllUsesWith(vBarg);

    // From now on, insertion point must point to the pipeline module body.
    // This ensures that pipeline stage instantiations and free-standing
    // operations are inserted into the pipeline module.
    builder.setInsertionPointToStart(pipelineMod.getBodyBlock());

    if (failed(runInner()))
      return failure();

    // Assign the pipeline module output op.
    auto pipelineOutputOp =
        cast<hw::OutputOp>(pipelineMod.getBodyBlock()->getTerminator());
    auto pipelineReturnOp =
        cast<pipeline::ReturnOp>(pipeline.getBodyBlock()->getTerminator());
    pipelineOutputOp->insertOperands(0, pipelineReturnOp.getInputs());

    pipeline.erase();
    return success();
  }

  void lowerStage(StageOp stage) override {
    OpBuilder::InsertionGuard guard(builder);
    auto [mod, inst] = buildStage(stage);

    // Move stage operations into the module.
    builder.setInsertionPointToStart(&mod.getBody().front());
    Value modClock = mod.getBody().front().getArgument(
        mod.getBody().front().getNumArguments() - 2);
    Value modReset = mod.getBody().front().getArgument(
        mod.getBody().front().getNumArguments() - 1);
    auto stageRets = emitStageBody(stage, modClock, modReset);

    // Assign the output operation to the stage return values.
    auto hwOutput = cast<hw::OutputOp>(mod.getBody().front().getTerminator());
    hwOutput->insertOperands(0, stageRets.regs);
    hwOutput->insertOperands(hwOutput.getNumOperands(), stageRets.passthroughs);
    hwOutput->insertOperands(hwOutput.getNumOperands(), {stageRets.valid});
  }

private:
  hw::HWModuleOp buildPipelineLike(Twine name, TypeRange inputs,
                                   TypeRange outputs, bool withEnableValid) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(parentModule);
    llvm::SmallVector<hw::PortInfo> ports;

    // Data inputs
    for (auto [idx, in] : llvm::enumerate(inputs))
      ports.push_back(
          hw::PortInfo{builder.getStringAttr("in" + std::to_string(idx)),
                       hw::PortDirection::INPUT, in});

    if (withEnableValid) {
      ports.push_back(hw::PortInfo{builder.getStringAttr("enable"),
                                   hw::PortDirection::INPUT,
                                   builder.getI1Type()});
    }

    // clock and reset
    ports.push_back(hw::PortInfo{builder.getStringAttr("clk"),
                                 hw::PortDirection::INPUT,
                                 builder.getI1Type()});
    ports.push_back(hw::PortInfo{builder.getStringAttr("rst"),
                                 hw::PortDirection::INPUT,
                                 builder.getI1Type()});

    for (auto [idx, out] : llvm::enumerate(outputs))
      ports.push_back(
          hw::PortInfo{builder.getStringAttr("out" + std::to_string(idx)),
                       hw::PortDirection::OUTPUT, out});

    if (withEnableValid) {
      ports.push_back(hw::PortInfo{builder.getStringAttr("valid"),
                                   hw::PortDirection::OUTPUT,
                                   builder.getI1Type()});
    }

    return builder.create<hw::HWModuleOp>(pipeline.getLoc(),
                                          builder.getStringAttr(name), ports);
  }

  std::tuple<hw::HWModuleOp, hw::InstanceOp> buildStage(StageOp stage) {
    builder.setInsertionPoint(parentModule);
    auto name = pipelineName + "_s" + std::to_string(stage.index());
    auto mod = buildPipelineLike(name, stage.getInputs().getTypes(),
                                 stage.getOutputs().getTypes(),
                                 /*withEnableValid=*/true);

    // instantiate...
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(pipelineMod.getBodyBlock()->getTerminator());
    llvm::SmallVector<Value, 4> stageOperands;
    llvm::append_range(stageOperands, stage.getOperands());
    stageOperands.push_back(pipelineClk);
    stageOperands.push_back(pipelineRst);
    auto inst = builder.create<hw::InstanceOp>(pipeline.getLoc(), mod,
                                               mod.getName(), stageOperands);

    // Remap the internal inputs of the stage to the module block arguments.
    for (auto [vInput, vBarg] :
         llvm::zip(stage.getInnerInputs(),
                   mod.getBody().front().getArguments().drop_back(3)))
      vInput.replaceAllUsesWith(vBarg);

    // And the valid...
    size_t validArgIdx = mod.getBody().front().getNumArguments() - 3;
    stage.getInnerEnable().replaceAllUsesWith(
        mod.getBody().front().getArgument(validArgIdx));

    // Replace the stage outputs with the instance results.
    for (auto [vOutput, vResult] :
         llvm::zip(stage.getResults(), inst.getResults()))
      vOutput.replaceAllUsesWith(vResult);

    return {mod, inst};
  }

  // Pipeline module clock.
  Value pipelineClk;
  // Pipeline module reset.
  Value pipelineRst;
  // Pipeline module, containing stage instantiations.
  hw::HWModuleOp pipelineMod;
};

//===----------------------------------------------------------------------===//
// Pipeline to HW Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PipelineToHWPass : public PipelineToHWBase<PipelineToHWPass> {
  void runOnOperation() override;
};

void PipelineToHWPass::runOnOperation() {
  OpBuilder builder(&getContext());
  // Iterate over each pipeline op in the module and convert.
  // Note: This pass matches on `hw::ModuleOp`s and not directly on the
  // `PipelineOp` due to the `PipelineOp` being erased during this pass.
  size_t pipelinesSeen = 0;
  for (auto pipeline :
       llvm::make_early_inc_range(getOperation().getOps<PipelineOp>())) {
    if (outlineStages) {
      if (failed(PipelineOutlineLowering(pipelinesSeen, pipeline, builder)
                     .run())) {
        signalPassFailure();
        return;
      }
    } else if (failed(PipelineInlineLowering(pipelinesSeen, pipeline, builder)
                          .run())) {
      signalPassFailure();
      return;
    }
    ++pipelinesSeen;
  }
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createPipelineToHWPass() {
  return std::make_unique<PipelineToHWPass>();
}
