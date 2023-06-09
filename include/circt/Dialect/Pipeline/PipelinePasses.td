//===-- PipelinePasses.td - Pipeline pass definition file --*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Pipeline dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_TD
#define CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_TD

include "mlir/Pass/PassBase.td"

def ExplicitRegs : Pass<"pipeline-explicit-regs", "pipeline::PipelineOp"> {
  let summary = "Makes stage registers explicit.";
  let description = [{
    Makes all stage-crossing def-use chains into explicit registers.
  }];
  let dependentDialects = ["hw::HWDialect"];
  let constructor = "circt::pipeline::createExplicitRegsPass()";
}

def StageSeparatorToStage : Pass<"pipeline-stagesep-to-stage", "pipeline::PipelineOp"> {
  let summary = "Lowers pipeline stage separations (`pipeline.stagesep`) into explicit stages.";
  let description = [{
    Lowers pipeline stage separations (`pipeline.stagesep`) into explicit stages.
  }];
  let dependentDialects = ["hw::HWDialect"];
  let constructor = "circt::pipeline::createStageSeparatorToStagePass()";
}


def ScheduleLinearPipeline : Pass<"pipeline-schedule-linear", "pipeline::PipelineOp"> {
  let summary = "Schedules a linear pipeline.";
  let description = [{
    Schedules a linear pipeline based on operator latencies.
  }];
  let dependentDialects = ["hw::HWDialect"];
  let constructor = "circt::pipeline::createScheduleLinearPipelinePass()";
}


#endif // CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_TD
