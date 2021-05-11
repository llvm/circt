//===- StaticLogicPasses.td - StaticLogic pass definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the StaticLogic
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef STATICLOGIC_PASSES
#define STATICLOGIC_PASSES

include "mlir/Pass/PassBase.td"

def SchedulePipeline : FunctionPass<"schedule-pipeline"> {
  let summary = "Schedule the operations in StaticLogic pipelines";
  let description = [{ TODO }];

  let constructor = "circt::staticlogic::createSchedulePipelinePass()";
}

#endif // STATICLOGIC_PASSES
