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

def ExplicitRegs : Pass<"pipeline-explicit-regs"> {
  let summary = "Makes stage registers explicit.";
  let description = [{
    Makes all stage-crossing def-use chains into explicit registers.
  }];
  let dependentDialects = ["hw::HWDialect"];
  let constructor = "circt::pipeline::createExplicitRegsPass()";
}

def ScheduleLinearPipeline : Pass<"pipeline-schedule-linear"> {
  let summary = "Schedules `pipeline.unscheduled` operations.";
  let description = [{
    Schedules `pipeline.unscheduled` operations based on operator latencies.
  }];
  let dependentDialects = ["hw::HWDialect", "seq::SeqDialect"];
  let constructor = "circt::pipeline::createScheduleLinearPipelinePass()";

  let options = [
    Option<"problemType", "problem-type", "std::string",
           /*default=*/"\"base\"",
           "Select between base problem and chaining problem ">,

    Option<"cycleTime", "cycle-time", "float",
           /*default=*/"1.0",
           "Cycle time, only applicable when chaining problem is selected ">,

    Option<"initInterval", "init-interval", "int",
        /*default=*/"1",
        "II">,  
  ];
}


#endif // CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_TD
