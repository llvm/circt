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

#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace circt;
using namespace pipeline;

// A class for generalizing pipeline lowering for both the inline and outlined
// implementation.
class PipelineLowering {
public:
  PipelineLowering(size_t pipelineID, ScheduledPipelineOp pipeline,
                   OpBuilder &builder)
      : pipelineID(pipelineID), pipeline(pipeline), builder(builder) {
    parentClk = pipeline.getClock();
    parentRst = pipeline.getReset();
    parentModule = pipeline->getParentOfType<hw::HWModuleOp>();
  }
  virtual ~PipelineLowering() = default;

  virtual LogicalResult run() = 0;

  struct StageReturns {
    llvm::SmallVector<Value> regs;
    llvm::SmallVector<Value> passthroughs;
  };

  virtual LogicalResult lowerStage(Block *stage, ValueRange stageArguments,
                                   size_t stageIndex) = 0;

  StageReturns emitStageBody(Block *stage, size_t stageIndex = -1,
                             Value clock = nullptr, Value reset = nullptr) {
    auto *terminator = stage->getTerminator();

    // Move the stage operations into the current insertion point.
    for (auto &op : llvm::make_early_inc_range(*stage)) {
      if (&op == terminator)
        continue;

      if (auto latencyOp = dyn_cast<LatencyOp>(op)) {
        // For now, just directly emit the body of the latency op. The latency
        // op is mainly used during register materialization. At a later stage,
        // we may want to add some TCL-related things here to communicate
        // multicycle paths.
        Block *latencyOpBody = latencyOp.getBodyBlock();
        for (auto &innerOp :
             llvm::make_early_inc_range(latencyOpBody->without_terminator()))
          innerOp.moveBefore(builder.getInsertionBlock(),
                             builder.getInsertionPoint());
        latencyOp.replaceAllUsesWith(
            latencyOpBody->getTerminator()->getOperands());
        latencyOp.erase();
      } else {
        op.moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
      }
    }

    StageReturns rets;
    auto stageTerminator = dyn_cast<StageOp>(terminator);
    if (!stageTerminator) {
      assert(isa<ReturnOp>(terminator) && "expected ReturnOp");
      // This was the pipeline return op - we're done.
      rets.passthroughs = terminator->getOperands();
      return rets;
    }

    // Build data registers
    auto stageRegPrefix = getStageRegPrefix(stageIndex);
    for (auto it : llvm::enumerate(stageTerminator.getRegisters())) {
      auto regIdx = it.index();
      auto regIn = it.value();
      auto regName = builder.getStringAttr(stageRegPrefix.strref() + "_reg" +
                                           std::to_string(regIdx));
      auto reg = builder.create<seq::CompRegOp>(
          stageTerminator->getLoc(), regIn.getType(), regIn, clock, regName,
          reset, Value(), StringAttr());
      rets.regs.push_back(reg);
    }

    rets.passthroughs = stageTerminator.getPassthroughs();
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
  ScheduledPipelineOp pipeline;

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

    // Replace uses of the external inputs with the inner external inputs.
    for (auto [outer, inner] :
         llvm::zip(pipeline.getExtInputs(), pipeline.getInnerExtInputs()))
      inner.replaceAllUsesWith(outer);

    // All operations should go directly before the pipeline op, into the
    // parent module.
    builder.setInsertionPoint(pipeline);
    if (failed(
            lowerStage(pipeline.getEntryStage(), pipeline.getInnerInputs(), 0)))
      return failure();

    return success();
  }

  /// NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult lowerStage(Block *stage, ValueRange stageArguments,
                           size_t stageIndex) override {
    OpBuilder::InsertionGuard guard(builder);

    if (stage != pipeline.getEntryStage()) {
      // Replace the internal stage inputs with the provided arguments.
      for (auto [vInput, vArg] :
           llvm::zip(pipeline.getStageArguments(stage), stageArguments))
        vInput.replaceAllUsesWith(vArg);
    }

    // Move stage operations into the current module.
    builder.setInsertionPoint(pipeline);
    StageReturns stageRets =
        emitStageBody(stage, stageIndex, parentClk, parentRst);

    if (auto nextStage = dyn_cast<StageOp>(stage->getTerminator())) {
      // Lower the next stage.
      SmallVector<Value> nextStageArgs;
      llvm::append_range(nextStageArgs, stageRets.regs);
      llvm::append_range(nextStageArgs, stageRets.passthroughs);
      return lowerStage(nextStage.getNextStage(), nextStageArgs,
                        stageIndex + 1);
    }

    // Replace the pipeline results with the return op operands.
    auto returnOp = cast<pipeline::ReturnOp>(stage->getTerminator());
    pipeline.replaceAllUsesWith(returnOp.getInputs());
    pipeline.erase();
    return success();
  }
};

class PipelineOutlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;

  StringAttr getStageRegPrefix(size_t stageIdx) override {
    return builder.getStringAttr("s" + std::to_string(stageIdx));
  }

  // Helper class to manage grabbing the various inputs for stage modules.
  struct PipelineStageMod {
    PipelineStageMod(PipelineOutlineLowering &parent, Block *block,
                     hw::HWModuleOp mod) {
      size_t nStageInputArgs = parent.pipeline.getNumStageArguments(block);
      inputs = mod.getArguments().take_front(nStageInputArgs);
      llvm::SmallVector<Value> stageExtInputs;
      parent.getStageExtInputs(block, stageExtInputs);
      extInputs =
          mod.getArguments().slice(nStageInputArgs, stageExtInputs.size());
      clock = mod.getArgument(mod.getNumArguments() - 2);
      reset = mod.getArgument(mod.getNumArguments() - 1);
    }

    ValueRange inputs;
    ValueRange extInputs;
    Value clock;
    Value reset;
  };

  // Iterate over the external inputs to the pipeline, and determine which
  // stages actually reference them. This will be used to generate the stage
  // module signatures.
  void gatherExtInputsToStages() {
    for (auto extIn : pipeline.getInnerExtInputs()) {
      for (auto *user : extIn.getUsers())
        stageExtInputs[user->getBlock()].insert(extIn);
    }
  }

  LogicalResult run() override {
    pipelineName =
        (parentModule.getName() + "_p" + std::to_string(pipelineID)).str();

    cloneConstantsToStages();

    // Build the top-level pipeline module.
    pipelineMod = buildPipelineLike(
        pipelineName, pipeline.getInputs().getTypes(),
        pipeline.getExtInputs().getTypes(), pipeline.getResults().getTypes());
    pipelineClk = pipelineMod.getBody().front().getArgument(
        pipelineMod.getBody().front().getNumArguments() - 2);
    pipelineRst = pipelineMod.getBody().front().getArgument(
        pipelineMod.getBody().front().getNumArguments() - 1);

    if (!pipeline.getExtInputs().empty()) {
      // Maintain a mapping between external inputs and their corresponding
      // block argument in the top-level pipeline.
      auto modInnerExtInputs =
          pipelineMod.getBody().front().getArguments().slice(
              pipeline.getExtInputs().getBeginOperandIndex(),
              pipeline.getExtInputs().size());
      for (auto [extIn, barg] :
           llvm::zip(pipeline.getInnerExtInputs(), modInnerExtInputs)) {
        toplevelExtInputs[extIn] = barg;
      }
    }

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

    // Determine the external inputs to each stage.
    gatherExtInputsToStages();

    // From now on, insertion point must point to the pipeline module body.
    // This ensures that pipeline stage instantiations and free-standing
    // operations are inserted into the pipeline module.
    builder.setInsertionPointToStart(pipelineMod.getBodyBlock());

    if (failed(lowerStage(
            pipeline.getEntryStage(),
            pipelineMod.getArguments().take_front(pipeline.getInputs().size()),
            0)))
      return failure();

    pipeline.erase();
    return success();
  }

  /// NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult lowerStage(Block *stage, ValueRange stageArguments,
                           size_t stageIndex) override {
    Block *nextStage = nullptr;
    Value modClock, modReset;
    hw::OutputOp stageOutputOp;
    ValueRange nextStageArgs;

    if (auto stageOp = dyn_cast<StageOp>(stage->getTerminator())) {
      auto [mod, inst] = buildStage(stage, stageArguments, stageIndex);
      auto pipelineStageMod = PipelineStageMod{*this, stage, mod};

      // Remap the internal inputs of the stage to the module block arguments.
      for (auto [vInput, vBarg] : llvm::zip(pipeline.getStageArguments(stage),
                                            pipelineStageMod.inputs))
        vInput.replaceAllUsesWith(vBarg);

      // Remap external inputs to the stage to the external inputs in the
      // module block arguments.
      llvm::SmallVector<Value> stageExtInputs;
      getStageExtInputs(stage, stageExtInputs);
      for (auto [vExtInput, vExtBarg] :
           llvm::zip(stageExtInputs, pipelineStageMod.extInputs)) {
        vExtInput.replaceUsesWithIf(vExtBarg, [&](OpOperand &operand) {
          return operand.getOwner()->getBlock() == stage;
        });
      }

      // Move stage operations into the module.
      builder.setInsertionPointToStart(&mod.getBody().front());
      modClock = pipelineStageMod.clock;
      modReset = pipelineStageMod.reset;
      stageOutputOp = cast<hw::OutputOp>(mod.getBody().front().getTerminator());
      nextStage = stageOp.getNextStage();
      nextStageArgs = inst.getResults();
    } else {
      // Remap the internal inputs of the stage to the stage arguments.
      for (auto [vInput, vBarg] :
           llvm::zip(pipeline.getStageArguments(stage), stageArguments))
        vInput.replaceAllUsesWith(vBarg);

      // Remap external inputs with the top-level pipeline module external
      // inputs.
      auto modInnerExtInputs = pipelineMod.getArguments().slice(
          pipeline.getInputs().size(), pipeline.getExtInputs().size());
      for (auto [extIn, topLevelBarg] :
           llvm::zip(pipeline.getInnerExtInputs(), modInnerExtInputs)) {
        extIn.replaceAllUsesWith(topLevelBarg);
      }

      stageOutputOp =
          cast<hw::OutputOp>(pipelineMod.getBodyBlock()->getTerminator());
      // Move lingering operations into the top-level pipeline module.
      builder.setInsertionPoint(stageOutputOp);
    }

    StageReturns stageRets =
        emitStageBody(stage, stageIndex, modClock, modReset);

    // Assign the output operation to the stage return values.
    stageOutputOp->insertOperands(0, stageRets.regs);
    stageOutputOp->insertOperands(stageOutputOp.getNumOperands(),
                                  stageRets.passthroughs);

    // Lower the next stage.
    if (nextStage)
      return lowerStage(nextStage, nextStageArgs, stageIndex + 1);

    return success();
  }

private:
  // Creates a clone of constant-like operations within each stage that
  // references them. This ensures that, once outlined, each stage will
  // reference valid constants.
  void cloneConstantsToStages() {
    // Maintain a mapping of the constants already cloned to a stage.
    for (auto &constantOp : llvm::make_early_inc_range(pipeline.getOps())) {
      if (!constantOp.hasTrait<OpTrait::ConstantLike>())
        continue;

      llvm::DenseMap<Block *, llvm::SmallVector<OpOperand *>> stageUses;
      Block *stageWithConstant = constantOp.getBlock();
      for (auto &use : constantOp.getUses()) {
        Block *usingStage = use.getOwner()->getBlock();
        if (usingStage == stageWithConstant)
          continue;
        stageUses[usingStage].push_back(&use);
      }

      // Clone the constant into each stage that uses it, and replace usages
      // within that stage.
      for (auto &[stage, uses] : stageUses) {
        Operation *clonedConstant = constantOp.clone();
        builder.setInsertionPointToStart(stage);
        builder.insert(clonedConstant);

        clonedConstant->setLoc(constantOp.getLoc());
        clonedConstant->moveBefore(&stage->front());
        for (OpOperand *use : uses)
          use->set(clonedConstant->getResult(0));
      }
    }
  }

  hw::HWModuleOp buildPipelineLike(Twine name, TypeRange inputs,
                                   TypeRange extInputs, TypeRange outputs) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(parentModule);
    llvm::SmallVector<hw::PortInfo> ports;

    // Data inputs
    for (auto [idx, in] : llvm::enumerate(inputs))
      ports.push_back(
          hw::PortInfo{builder.getStringAttr("in" + std::to_string(idx)),
                       hw::PortDirection::INPUT, in});

    // External inputs
    for (auto [idx, in] : llvm::enumerate(extInputs))
      ports.push_back(
          hw::PortInfo{builder.getStringAttr("extIn" + std::to_string(idx)),
                       hw::PortDirection::INPUT, in});

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

    return builder.create<hw::HWModuleOp>(pipeline.getLoc(),
                                          builder.getStringAttr(name), ports);
  }

  std::tuple<hw::HWModuleOp, hw::InstanceOp>
  buildStage(Block *stage, ValueRange stageArguments, size_t stageIndex) {
    builder.setInsertionPoint(parentModule);
    auto name = pipelineName + "_s" + std::to_string(stageIndex);

    llvm::SmallVector<Type> outputTypes;
    if (auto stageOp = dyn_cast<StageOp>(stage->getTerminator()))
      llvm::append_range(
          outputTypes,
          pipeline.getStageArguments(stageOp.getNextStage()).getTypes());
    else
      llvm::append_range(outputTypes, pipeline.getResultTypes());

    llvm::SmallVector<Value> stageExtInputs;
    getStageExtInputs(stage, stageExtInputs);
    hw::HWModuleOp mod =
        buildPipelineLike(name, pipeline.getStageArguments(stage).getTypes(),
                          ValueRange(stageExtInputs).getTypes(), outputTypes);

    // instantiate...
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(pipelineMod.getBodyBlock()->getTerminator());
    llvm::SmallVector<Value, 4> stageOperands;
    llvm::append_range(stageOperands, stageArguments);

    // Gather external inputs for this stage from the top-level pipeline
    // module.
    for (auto extInput : stageExtInputs)
      stageOperands.push_back(toplevelExtInputs[extInput]);

    stageOperands.push_back(pipelineClk);
    stageOperands.push_back(pipelineRst);
    auto inst = builder.create<hw::InstanceOp>(pipeline.getLoc(), mod,
                                               mod.getName(), stageOperands);

    return {mod, inst};
  }

  // Pipeline module clock.
  Value pipelineClk;
  // Pipeline module reset.
  Value pipelineRst;
  // Pipeline module, containing stage instantiations.
  hw::HWModuleOp pipelineMod;

  // Handle to the instantiation of the last stage in the pipeline.
  hw::InstanceOp lastStageInst;

  // A mapping between stages and the external inputs which they reference.
  // A SetVector is used to ensure determinism in the order of the external
  // inputs to a stage.
  llvm::DenseMap<Block *, llvm::SetVector<Value>> stageExtInputs;

  // A mapping between external inputs and their corresponding top-level
  // input in the pipeline module.
  llvm::DenseMap<Value, Value> toplevelExtInputs;

  // Wrapper around stageExtInputs which returns a llvm::SmallVector<Value>,
  // which in turn can be used to get a ValueRange (and by extension,
  // TypeRange).
  void getStageExtInputs(Block *stage, llvm::SmallVector<Value> &extInputs) {
    llvm::append_range(extInputs, stageExtInputs[stage]);
  }
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
  // `ScheduledPipelineOp` due to the `ScheduledPipelineOp` being erased during
  // this pass.
  size_t pipelinesSeen = 0;
  for (auto pipeline : llvm::make_early_inc_range(
           getOperation().getOps<ScheduledPipelineOp>())) {
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
