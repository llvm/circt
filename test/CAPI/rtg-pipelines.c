//===- rtg-pipelines.c - RTG pipeline CAPI unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: circt-capi-rtg-pipelines-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "circt-c/Dialect/RTG.h"
#include "circt-c/RtgTool.h"

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__rtg__(), ctx);

  MlirModule moduleOp = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(
               "rtg.sequence @seq() {\n"
               "}\n"
               "rtg.test @test() {\n"
               "  %0 = rtg.get_sequence @seq : !rtg.sequence\n"
               "}\n"));
  if (mlirModuleIsNull(moduleOp)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }

  MlirPassManager pm = mlirPassManagerCreate(ctx);
  CirctRtgToolOptions options = circtRtgToolOptionsCreateDefault(/*seed=*/0);
  circtRtgToolRandomizerPipeline(pm, options);

  MlirOperation op = mlirModuleGetOperation(moduleOp);

  if (mlirLogicalResultIsFailure(mlirPassManagerRunOnOp(pm, op))) {
    printf("ERROR: Pipeline run failed.\n");
    mlirModuleDestroy(moduleOp);
    mlirPassManagerDestroy(pm);
    circtRtgToolOptionsDestroy(options);
    mlirContextDestroy(ctx);
    return 1;
  }

  // CHECK-LABEL: rtg.test @test
  // CHECK-NEXT:  }
  mlirOperationDump(op);

  mlirModuleDestroy(moduleOp);
  mlirPassManagerDestroy(pm);
  circtRtgToolOptionsDestroy(options);
  mlirContextDestroy(ctx);
  return 0;
}
