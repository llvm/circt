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
  for (auto pipeline :
       llvm::make_early_inc_range(getOperation().getOps<PipelineOp>()))
    if (failed(lowerPipeline(pipeline, builder)))
      signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass> circt::createPipelineToHWPass() {
  return std::make_unique<PipelineToHWPass>();
}
