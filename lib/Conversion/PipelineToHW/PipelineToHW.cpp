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
#include "circt/Support/ValueMapper.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace pipeline;

// A class for generalizing pipeline lowering for both inline and outlined
// implementation. This revolves around usage of a ValueMapper (replacing the
// more traditional Value::replaceAllUsesWith(...)) approach, to allow for
// maintaining temporary value mappings across module/scope boundaries.
class PipelineLowering {
public:
  PipelineLowering(size_t pipelineID, PipelineOp pipeline, OpBuilder &builder)
      : pipelineID(pipelineID), pipeline(pipeline), builder(builder) {}
  virtual ~PipelineLowering() = default;

  LogicalResult run() {
    clk = pipeline.getClock();
    rst = pipeline.getReset();

    // Handle the entry stage;
    currentStage = getNextStage();
    currentStageIndex = 0;
    createCurrentStage(pipeline.getInnerInputs(), pipeline.getGo());

    // Iterate over the pipeline operations,
    for (auto &op : llvm::make_early_inc_range(*pipeline.getBodyBlock())) {
      auto loc = op.getLoc();
      llvm::TypeSwitch<Operation *, void>(&op)
          .Case<PipelineStageRegisterOp>([&](auto stage) {
            assert(currentStage == stage && "Unexpected stage order");

            auto [currClk, currRst] = getCurrentClockAndReset();

            unsigned stageIdx = stage.index();
            auto validRegName = builder.getStringAttr(
                "s" + std::to_string(stageIdx) + "_valid");
            auto validReg = builder.create<seq::CompRegOp>(
                loc, builder.getI1Type(), stage.getWhen(), currClk,
                validRegName, currRst, Value(), StringAttr());
            stage.getValid().replaceAllUsesWith(validReg);

            for (auto it : llvm::enumerate(stage.getRegIns())) {
              auto regIdx = it.index();
              auto regIn = it.value();
              auto regName =
                  builder.getStringAttr("s" + std::to_string(stageIdx) +
                                        "_reg" + std::to_string(regIdx));
              auto reg = builder.create<seq::CompRegOp>(
                  loc, regIn.getType(), regIn, currClk, regName, currRst,
                  Value(), StringAttr());
              stage.getRegOuts()[regIdx].replaceAllUsesWith(reg);
            }

            // Finalize the current stage and create the next stage.
            finalizeCurrentStage(stage.getOperands());
            currentStage = getNextStage();
            ++currentStageIndex;
            createCurrentStage(stage.getRegOuts(), stage.getValid());
          })
          .Case<pipeline::ReturnOp>([&](auto ret) {
            finalizeCurrentStage(ret.getOperands());
            handle(ret);
          })
          .Default([&](auto op) {
            // Move the operation into the current insertion point.
            op->moveBefore(builder.getInsertionBlock(),
                           builder.getInsertionPoint());
          });
    }

    pipeline->getParentOfType<mlir::ModuleOp>().dump();
    pipeline->replaceAllUsesWith(getReturnValues());

    pipeline.erase();
    return success();
  }

  // A dispatch function for handling the conversion state transfer to the next
  // state op.
  virtual void finalizeCurrentStage(ValueRange stageOperands) = 0;

  // A dispatch function for handling the creation of the current stage.
  // valueIns: The inputs to the stage.
  // validIn:  The valid signal to the stage.
  virtual void createCurrentStage(ValueRange valueIns, Value validIn) = 0;

  // Returns a reference to the clock and reset values of the current scope.
  virtual std::pair<Value, Value> getCurrentClockAndReset() {
    return {clk, rst};
  }

  // Returns the next stage wrt. the current stage, if any.
  PipelineStageRegisterOp getNextStage() {
    auto stageOps = pipeline.getOps<PipelineStageRegisterOp>();
    // Edge case - no stages
    if (stageOps.empty())
      return nullptr;

    if (!currentStage)
      return *stageOps.begin();

    auto it = llvm::find(stageOps, currentStage);
    assert(it != stageOps.end() && "Could not find current stage in pipeline");
    // Is it the last stage?
    if (std::next(it) == stageOps.end())
      return nullptr;

    return *std::next(it);
  }

  // Dispatch function for handling the return op - this is
  // implementation-specific.
  virtual void handle(pipeline::ReturnOp) = 0;

  // Function to retrieve the return values of the pipeline, which will replace
  // the original pipeline returns.
  virtual ValueRange getReturnValues() = 0;

protected:
  // A handle to the current stage op being converted.
  PipelineStageRegisterOp currentStage;

  // Pipeline clock.
  Value clk;
  // Pipeline reset.
  Value rst;
  // ID of the current pipeline, used for naming.
  size_t pipelineID;
  // The current pipeline to be converted.
  PipelineOp pipeline;

  OpBuilder &builder;

  // A value mapper to maintain mappings of values from within the
  // pipeline to the lowered pipeline.
  ValueMapper mapper;

  // The index of the current stage.
  size_t currentStageIndex = 0;
};

class PipelineInlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;
  void handle(pipeline::ReturnOp ret) override { retOp = ret; }

  ValueRange getReturnValues() override {
    // Return the operands of the return op.
    return retOp.getOutputs();
  }

  void finalizeCurrentStage(ValueRange stageOperands) override {}

  void createCurrentStage(ValueRange inputs, Value valid) override {}

private:
  // Store a reference to the return op, so that we have a shorthand way of
  // accessing the return values of the pipeline.
  pipeline::ReturnOp retOp;
};

class PipelineOutlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;
  void handle(pipeline::ReturnOp ret) override {}

  ValueRange getReturnValues() override {
    // Return the outputs of the final stage instantiation.
    return currentStageInst.getResults().drop_back(1);
  }

  void finalizeCurrentStage(ValueRange inputs) override {
    // Set the current module output operands to the inputs of the stage op.
    auto outputOp =
        cast<hw::OutputOp>(currentStageMod.getBody().front().getTerminator());
    outputOp->insertOperands(0, inputs);

    if (currentStage) {
      // Remap the return values of the stage op to map to the return values of
      // the stage instance. These will later be remapped (again) to map
      // internally in the next stage module.
      for (auto [resStage, resInst] :
           llvm::zip(currentStage.getResults(), currentStageInst.getResults()))
        resStage.replaceAllUsesWith(resInst);
    }
  }

  void createCurrentStage(ValueRange inputs, Value valid) override {
    auto parent = pipeline->getParentOfType<hw::HWModuleOp>();
    builder.setInsertionPoint(parent);
    auto name = builder.getStringAttr(parent.getName() + "_p" +
                                      std::to_string(pipelineID) + "_s" +
                                      std::to_string(currentStageIndex));
    llvm::SmallVector<hw::PortInfo> ports;

    // Data inputs
    for (auto [idx, in] : llvm::enumerate(inputs.getTypes()))
      ports.push_back(
          hw::PortInfo{builder.getStringAttr("in" + std::to_string(idx)),
                       hw::PortDirection::INPUT, in});

    // Enable input
    ports.push_back(hw::PortInfo{builder.getStringAttr("enable"),
                                 hw::PortDirection::INPUT,
                                 builder.getI1Type()});

    // The output types of the current stage will be the input types to the next
    // stage.
    for (auto [idx, out] : llvm::enumerate(getNextStageInputTypes()))
      ports.push_back(
          hw::PortInfo{builder.getStringAttr("out" + std::to_string(idx)),
                       hw::PortDirection::OUTPUT, out});

    // Valid output
    ports.push_back(hw::PortInfo{builder.getStringAttr("valid"),
                                 hw::PortDirection::OUTPUT,
                                 builder.getI1Type()});

    // clock and reset
    ports.push_back(hw::PortInfo{builder.getStringAttr("clk"),
                                 hw::PortDirection::INPUT,
                                 builder.getI1Type()});
    ports.push_back(hw::PortInfo{builder.getStringAttr("rst"),
                                 hw::PortDirection::INPUT,
                                 builder.getI1Type()});

    currentStageMod =
        builder.create<hw::HWModuleOp>(pipeline.getLoc(), name, ports);

    // instantiate...
    builder.setInsertionPoint(pipeline);
    llvm::SmallVector<Value, 4> stageOperands;
    llvm::append_range(stageOperands, inputs);
    stageOperands.push_back(clk);
    stageOperands.push_back(rst);
    currentStageInst = builder.create<hw::InstanceOp>(
        pipeline.getLoc(), currentStageMod, currentStageMod.getName(),
        stageOperands);

    // Set insertion point to the stage module, making all operations go into
    // it.
    builder.setInsertionPoint(&currentStageMod.getBody().front().front());

    // Remap the inputs to the stage instantiation to map to the block arguments
    // within the stage.
    for (auto [vInput, vBarg] : llvm::zip(
             inputs,
             currentStageMod.getBody().front().getArguments().drop_back(3)))
      vInput.replaceAllUsesWith(vBarg);

    // And the valid...
    valid.replaceAllUsesWith(currentStageMod.getBody().front().getArgument(
        currentStageMod.getBody().front().getNumArguments() - 3));
  }

  std::pair<Value, Value> getCurrentClockAndReset() override {
    auto &entryBlock = currentStageMod.getBody().front();
    Value modClk = entryBlock.getArgument(entryBlock.getNumArguments() - 2);
    Value modRst = entryBlock.getArgument(entryBlock.getNumArguments() - 1);
    return {modClk, modRst};
  }

  // Returns the input values to the next stage (or return) op.
  TypeRange getNextStageInputTypes() {
    if (auto nextStage = getNextStage())
      return nextStage.getOperands().drop_back(1).getTypes();

    auto *terminator = pipeline.getBodyBlock()->getTerminator();
    return terminator->getOperands().drop_back(1).getTypes();
  }

private:
  // A reference to the module corresponding to the current stage under
  // conversion.
  hw::HWModuleOp currentStageMod;
  // A reference to the instantiation of the current stage module.
  hw::InstanceOp currentStageInst;
};

static LogicalResult lowerPipeline(PipelineOp pipeline, OpBuilder &builder) {
  if (pipeline.isLatencyInsensitive())
    return pipeline.emitOpError() << "Only latency-sensitive pipelines are "
                                     "supported at the moment";

  // Simply move the ops from the pipeline to the enclosing hw.module scope,
  // converting any stage ops to seq registers.
  Value clk = pipeline.getClock();
  Value rst = pipeline.getReset();
  llvm::SmallVector<Value, 4> retVals;
  builder.setInsertionPoint(pipeline);

  for (auto [arg, barg] : llvm::zip(pipeline.getOperands(),
                                    pipeline.getBodyBlock()->getArguments()))
    barg.replaceAllUsesWith(arg);

  for (auto &op : llvm::make_early_inc_range(*pipeline.getBodyBlock())) {
    auto loc = op.getLoc();
    llvm::TypeSwitch<Operation *, void>(&op)
        .Case<PipelineStageRegisterOp>([&](auto stage) {
          unsigned stageIdx = stage.index();
          auto validRegName =
              builder.getStringAttr("s" + std::to_string(stageIdx) + "_valid");
          auto validReg = builder.create<seq::CompRegOp>(
              loc, builder.getI1Type(), stage.getWhen(), clk, validRegName, rst,
              Value(), StringAttr());
          stage.getValid().replaceAllUsesWith(validReg);

          for (auto it : llvm::enumerate(stage.getRegIns())) {
            auto regIdx = it.index();
            auto regIn = it.value();
            auto regName =
                builder.getStringAttr("s" + std::to_string(stageIdx) + "_reg" +
                                      std::to_string(regIdx));
            auto reg = builder.create<seq::CompRegOp>(loc, regIn.getType(),
                                                      regIn, clk, regName, rst,
                                                      Value(), StringAttr());
            stage.getRegOuts()[regIdx].replaceAllUsesWith(reg);
          }
        })
        .Case<pipeline::ReturnOp>([&](auto ret) { retVals = ret.getOutputs(); })
        .Default([&](auto op) { op->moveBefore(pipeline); });
  }

  pipeline->replaceAllUsesWith(retVals);
  pipeline.erase();
  return success();
}

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
  size_t count = 0;
  for (auto pipeline :
       llvm::make_early_inc_range(getOperation().getOps<PipelineOp>())) {
    if (outlineStages &&
        failed(PipelineOutlineLowering(count, pipeline, builder).run())) {
      signalPassFailure();
      return;
    } else if (failed(PipelineInlineLowering(count, pipeline, builder).run())) {
      signalPassFailure();
      return;
    }
    count++;
  }
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createPipelineToHWPass() {
  return std::make_unique<PipelineToHWPass>();
}
