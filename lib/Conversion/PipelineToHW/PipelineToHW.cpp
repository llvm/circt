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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Pipeline/PipelineOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_PIPELINETOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace pipeline;

namespace {
// Base class for all pipeline lowerings.
class PipelineLowering {
public:
  PipelineLowering(size_t pipelineID, ScheduledPipelineOp pipeline,
                   OpBuilder &builder, bool clockGateRegs,
                   bool enablePowerOnValues)
      : pipelineID(pipelineID), pipeline(pipeline), builder(builder),
        clockGateRegs(clockGateRegs), enablePowerOnValues(enablePowerOnValues) {
    parentClk = pipeline.getClock();
    parentRst = pipeline.getReset();
    parentModule = pipeline->getParentOfType<hw::HWModuleOp>();
  }
  virtual ~PipelineLowering() = default;

  virtual LogicalResult run() = 0;

  // Arguments used for emitting the body of a stage module. These values must
  // be within the scope of the stage module body.
  struct StageArgs {
    ValueRange data;
    Value enable;
    Value stall;
    Value clock;
    Value reset;
    Value lnsEn;
  };

  // Arguments used for returning the results from a stage. These values must
  // be within the scope of the stage module body.
  struct StageReturns {
    llvm::SmallVector<Value> regs;
    llvm::SmallVector<Value> passthroughs;
    Value valid;

    // In case this was the last register in a non-stallable register chain, the
    // register will also return its enable signal to be used for LNS of
    // downstream stages.
    Value lnsEn;
  };

  virtual FailureOr<StageReturns>
  lowerStage(Block *stage, StageArgs args, size_t stageIndex,
             llvm::ArrayRef<Attribute> inputNames = {}) = 0;

  StageReturns emitStageBody(Block *stage, StageArgs args,
                             llvm::ArrayRef<Attribute> registerNames,
                             size_t stageIndex = -1) {
    assert(args.enable && "enable not set");
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

    auto loc = terminator->getLoc();
    Value notStalled;
    auto getOrSetNotStalled = [&]() {
      if (!notStalled) {
        notStalled = comb::createOrFoldNot(loc, args.stall, builder);
      }
      return notStalled;
    };

    // Determine the stage kind. This will influence how the stage valid and
    // enable signals are defined.
    StageKind stageKind = pipeline.getStageKind(stageIndex);
    Value stageValid;
    StringAttr validSignalName =
        builder.getStringAttr(getStagePrefix(stageIndex).strref() + "_valid");
    switch (stageKind) {
    case StageKind::Continuous:
      LLVM_FALLTHROUGH;
    case StageKind::NonStallable:
      stageValid = args.enable;
      break;
    case StageKind::Stallable:
      stageValid =
          builder.create<comb::AndOp>(loc, args.enable, getOrSetNotStalled());
      stageValid.getDefiningOp()->setAttr("sv.namehint", validSignalName);
      break;
    case StageKind::Runoff:
      assert(args.lnsEn && "Expected an LNS signal if this was a runoff stage");
      stageValid = builder.create<comb::AndOp>(
          loc, args.enable,
          builder.create<comb::OrOp>(loc, args.lnsEn, getOrSetNotStalled()));
      stageValid.getDefiningOp()->setAttr("sv.namehint", validSignalName);
      break;
    }

    StageReturns rets;
    auto stageOp = dyn_cast<StageOp>(terminator);
    if (!stageOp) {
      assert(isa<ReturnOp>(terminator) && "expected ReturnOp");
      // This was the pipeline return op - the return op/last stage doesn't
      // register its operands, hence, all return operands are passthrough
      // and the valid signal is equal to the unregistered enable signal.
      rets.passthroughs = terminator->getOperands();
      rets.valid = stageValid;
      return rets;
    }

    assert(registerNames.size() == stageOp.getRegisters().size() &&
           "register names and registers must be the same size");

    bool isStallablePipeline = stageKind != StageKind::Continuous;
    Value notStalledClockGate;
    if (this->clockGateRegs) {
      // Create the top-level clock gate.
      notStalledClockGate = builder.create<seq::ClockGateOp>(
          loc, args.clock, stageValid, /*test_enable=*/Value(),
          /*inner_sym=*/hw::InnerSymAttr());
    }

    for (auto it : llvm::enumerate(stageOp.getRegisters())) {
      auto regIdx = it.index();
      auto regIn = it.value();

      StringAttr regName = cast<StringAttr>(registerNames[regIdx]);
      Value dataReg;
      if (this->clockGateRegs) {
        // Use the clock gate instead of clock enable.
        Value currClockGate = notStalledClockGate;
        for (auto hierClockGateEnable : stageOp.getClockGatesForReg(regIdx)) {
          // Create clock gates for any hierarchically nested clock gates.
          currClockGate = builder.create<seq::ClockGateOp>(
              loc, currClockGate, hierClockGateEnable,
              /*test_enable=*/Value(),
              /*inner_sym=*/hw::InnerSymAttr());
        }
        dataReg = builder.create<seq::CompRegOp>(stageOp->getLoc(), regIn,
                                                 currClockGate, regName);
      } else {
        // Only clock-enable the register if the pipeline is stallable.
        // For non-stallable (continuous) pipelines, a data register can always
        // be clocked.
        if (isStallablePipeline) {
          dataReg = builder.create<seq::CompRegClockEnabledOp>(
              stageOp->getLoc(), regIn, args.clock, stageValid, regName);
        } else {
          dataReg = builder.create<seq::CompRegOp>(stageOp->getLoc(), regIn,
                                                   args.clock, regName);
        }
      }
      rets.regs.push_back(dataReg);
    }

    rets.valid = stageValid;
    if (stageKind == StageKind::NonStallable)
      rets.lnsEn = args.enable;

    rets.passthroughs = stageOp.getPassthroughs();
    return rets;
  }

  // A container carrying all-things stage output naming related.
  // To avoid overloading 'output's to much (i'm trying to keep that
  // reserved for "output" ports), this is named "egress".
  struct StageEgressNames {
    llvm::SmallVector<Attribute> regNames;
    llvm::SmallVector<Attribute> outNames;
    llvm::SmallVector<Attribute> inNames;
  };

  // Returns a set of names for the output values of a given stage
  // (registers and passthrough). If `withPipelinePrefix` is true, the names
  // will be prefixed with the pipeline name.
  void getStageEgressNames(size_t stageIndex, Operation *stageTerminator,
                           bool withPipelinePrefix,
                           StageEgressNames &egressNames) {
    StringAttr pipelineName;
    if (withPipelinePrefix)
      pipelineName = getPipelineBaseName();

    if (auto stageOp = dyn_cast<StageOp>(stageTerminator)) {
      // Registers...
      std::string assignedRegName, assignedOutName, assignedInName;
      for (size_t regi = 0; regi < stageOp.getRegisters().size(); ++regi) {
        if (auto regName = stageOp.getRegisterName(regi)) {
          assignedRegName = regName.str();
          assignedOutName = assignedRegName + "_out";
          assignedInName = assignedRegName + "_in";
        } else {
          assignedRegName =
              ("stage" + Twine(stageIndex) + "_reg" + Twine(regi)).str();
          assignedOutName = ("out" + Twine(regi)).str();
          assignedInName = ("in" + Twine(regi)).str();
        }

        if (pipelineName && !pipelineName.getValue().empty()) {
          assignedRegName = pipelineName.str() + "_" + assignedRegName;
          assignedOutName = pipelineName.str() + "_" + assignedOutName;
          assignedInName = pipelineName.str() + "_" + assignedInName;
        }

        egressNames.regNames.push_back(builder.getStringAttr(assignedRegName));
        egressNames.outNames.push_back(builder.getStringAttr(assignedOutName));
        egressNames.inNames.push_back(builder.getStringAttr(assignedInName));
      }

      // Passthroughs
      for (size_t passi = 0; passi < stageOp.getPassthroughs().size();
           ++passi) {
        if (auto passName = stageOp.getPassthroughName(passi)) {
          assignedOutName = (passName.strref() + "_out").str();
          assignedInName = (passName.strref() + "_in").str();
        } else {
          assignedOutName = ("pass" + Twine(passi)).str();
          assignedInName = ("pass" + Twine(passi)).str();
        }

        if (pipelineName && !pipelineName.getValue().empty()) {
          assignedOutName = pipelineName.str() + "_" + assignedOutName;
          assignedInName = pipelineName.str() + "_" + assignedInName;
        }

        egressNames.outNames.push_back(builder.getStringAttr(assignedOutName));
        egressNames.inNames.push_back(builder.getStringAttr(assignedInName));
      }
    } else {
      // For the return op, we just inherit the names of the top-level
      // pipeline as stage output names.
      llvm::copy(pipeline.getOutputNames().getAsRange<StringAttr>(),
                 std::back_inserter(egressNames.outNames));
    }
  }

  // Returns a string to be used as a prefix for all stage registers.
  virtual StringAttr getStagePrefix(size_t stageIdx) = 0;

protected:
  // Determine a reasonable name for the pipeline. This will affect naming
  // of things such as stage registers.
  StringAttr getPipelineBaseName() {
    if (auto nameAttr = pipeline.getNameAttr())
      return nameAttr;
    return StringAttr::get(pipeline.getContext(), "p" + Twine(pipelineID));
  }

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

  // If true, will use clock gating for registers instead of input muxing.
  bool clockGateRegs;

  // If true, will add power-on values to the control registers of the design.
  bool enablePowerOnValues;

  // Name of this pipeline - used for naming stages and registers.
  // Implementation defined.
  StringAttr pipelineName;
};

class PipelineInlineLowering : public PipelineLowering {
public:
  using PipelineLowering::PipelineLowering;

  StringAttr getStagePrefix(size_t stageIdx) override {
    if (pipelineName && !pipelineName.getValue().empty())
      return builder.getStringAttr(pipelineName.strref() + "_stage" +
                                   Twine(stageIdx));
    return builder.getStringAttr("stage" + Twine(stageIdx));
  }

  LogicalResult run() override {
    pipelineName = getPipelineBaseName();

    // Replace uses of the pipeline internal inputs with the pipeline inputs.
    for (auto [outer, inner] :
         llvm::zip(pipeline.getInputs(), pipeline.getInnerInputs()))
      inner.replaceAllUsesWith(outer);

    // All operations should go directly before the pipeline op, into the
    // parent module.
    builder.setInsertionPoint(pipeline);
    StageArgs args;
    args.data = pipeline.getInnerInputs();
    args.enable = pipeline.getGo();
    args.clock = pipeline.getClock();
    args.reset = pipeline.getReset();
    args.stall = pipeline.getStall();
    if (failed(lowerStage(pipeline.getEntryStage(), args, 0)))
      return failure();

    pipeline.erase();
    return success();
  }

  /// NOLINTNEXTLINE(misc-no-recursion)
  FailureOr<StageReturns>
  lowerStage(Block *stage, StageArgs args, size_t stageIndex,
             llvm::ArrayRef<Attribute> /*inputNames*/ = {}) override {
    OpBuilder::InsertionGuard guard(builder);
    Operation *terminator = stage->getTerminator();
    Location loc = terminator->getLoc();

    if (stage != pipeline.getEntryStage()) {
      // Replace the internal stage inputs with the provided arguments.
      for (auto [vInput, vArg] :
           llvm::zip(pipeline.getStageDataArgs(stage), args.data))
        vInput.replaceAllUsesWith(vArg);
    }

    // Build stage enable register. The enable register is reset to 0 iff a
    // reset signal is available. We here rely on the compreg builders, which
    // accept reset signal/reset value mlir::Value's that are null.
    //
    // The stage enable register takes the
    // previous-stage combinational valid output and determines whether this
    // stage is active or not in the next cycle. A non-stallable stage always
    // registers the incoming enable signal, whereas other stages register based
    // on the current stall state.
    StageKind stageKind = pipeline.getStageKind(stageIndex);
    Value stageEnabled;
    if (stageIndex == 0) {
      stageEnabled = args.enable;
    } else {
      auto stageRegPrefix = getStagePrefix(stageIndex);
      auto enableRegName = (stageRegPrefix.strref() + "_enable").str();

      Value enableRegResetVal;
      if (args.reset)
        enableRegResetVal =
            builder.create<hw::ConstantOp>(loc, APInt(1, 0, false)).getResult();

      switch (stageKind) {
      case StageKind::Continuous:
        LLVM_FALLTHROUGH;
      case StageKind::NonStallable:
        stageEnabled = builder.create<seq::CompRegOp>(
            loc, args.enable, args.clock, args.reset, enableRegResetVal,
            enableRegName);
        break;
      case StageKind::Stallable:
        stageEnabled = builder.create<seq::CompRegClockEnabledOp>(
            loc, args.enable, args.clock,
            comb::createOrFoldNot(loc, args.stall, builder), args.reset,
            enableRegResetVal, enableRegName);
        break;
      case StageKind::Runoff:
        assert(args.lnsEn &&
               "Expected an LNS signal if this was a runoff stage");
        stageEnabled = builder.create<seq::CompRegClockEnabledOp>(
            loc, args.enable, args.clock,
            builder.create<comb::OrOp>(
                loc, args.lnsEn,
                comb::createOrFoldNot(loc, args.stall, builder)),
            args.reset, enableRegResetVal, enableRegName);
        break;
      }

      if (enablePowerOnValues) {
        llvm::TypeSwitch<Operation *, void>(stageEnabled.getDefiningOp())
            .Case<seq::CompRegOp, seq::CompRegClockEnabledOp>([&](auto op) {
              op.getInitialValueMutable().assign(
                  circt::seq::createConstantInitialValue(
                      builder, loc,
                      builder.getIntegerAttr(builder.getI1Type(),
                                             APInt(1, 0, false))));
            });
      }
    }

    // Replace the stage valid signal.
    args.enable = stageEnabled;
    pipeline.getStageEnableSignal(stage).replaceAllUsesWith(stageEnabled);

    // Determine stage egress info.
    auto nextStage = dyn_cast<StageOp>(terminator);
    StageEgressNames egressNames;
    if (nextStage)
      getStageEgressNames(stageIndex, nextStage,
                          /*withPipelinePrefix=*/true, egressNames);

    // Move stage operations into the current module.
    builder.setInsertionPoint(pipeline);
    StageReturns stageRets =
        emitStageBody(stage, args, egressNames.regNames, stageIndex);

    if (nextStage) {
      // Lower the next stage.
      SmallVector<Value> nextStageArgs;
      llvm::append_range(nextStageArgs, stageRets.regs);
      llvm::append_range(nextStageArgs, stageRets.passthroughs);
      args.enable = stageRets.valid;
      if (stageRets.lnsEn) {
        // Swap the lnsEn signal if the current stage lowering generated an
        // lnsEn.
        args.lnsEn = stageRets.lnsEn;
      }
      args.data = nextStageArgs;
      return lowerStage(nextStage.getNextStage(), args, stageIndex + 1);
    }

    // Replace the pipeline results with the return op operands.
    auto returnOp = cast<pipeline::ReturnOp>(stage->getTerminator());
    llvm::SmallVector<Value> pipelineReturns;
    llvm::append_range(pipelineReturns, returnOp.getInputs());
    // The last stage valid signal is the 'done' output of the pipeline.
    pipelineReturns.push_back(stageRets.valid);
    pipeline.replaceAllUsesWith(pipelineReturns);
    return stageRets;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pipeline to HW Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PipelineToHWPass
    : public circt::impl::PipelineToHWBase<PipelineToHWPass> {
  using PipelineToHWBase::PipelineToHWBase;
  void runOnOperation() override;

private:
  // Lowers pipelines within HWModules. This pass is currently expecting that
  // Pipelines are always nested with HWModule's but could be written to be
  // more generic.
  void runOnHWModule(hw::HWModuleOp mod);
};

void PipelineToHWPass::runOnOperation() {
  for (auto hwMod : getOperation().getOps<hw::HWModuleOp>())
    runOnHWModule(hwMod);
}

void PipelineToHWPass::runOnHWModule(hw::HWModuleOp mod) {
  OpBuilder builder(&getContext());
  // Iterate over each pipeline op in the module and convert.
  // Note: This pass matches on `hw::ModuleOp`s and not directly on the
  // `ScheduledPipelineOp` due to the `ScheduledPipelineOp` being erased
  // during this pass.
  size_t pipelinesSeen = 0;
  for (auto pipeline :
       llvm::make_early_inc_range(mod.getOps<ScheduledPipelineOp>())) {
    if (failed(PipelineInlineLowering(pipelinesSeen, pipeline, builder,
                                      clockGateRegs, enablePowerOnValues)
                   .run())) {
      signalPassFailure();
      return;
    }
    ++pipelinesSeen;
  }
}

} // namespace

std::unique_ptr<mlir::Pass>
circt::createPipelineToHWPass(const PipelineToHWOptions &options) {
  return std::make_unique<PipelineToHWPass>(options);
}
