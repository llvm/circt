/*===- firtool.c - Simple test of FIRRTL C APIs ---------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-firtool-test 2>&1 | FileCheck %s
 */

#include "circt-c/Firtool/Firtool.h"
#include "circt-c/Dialect/FIRRTL.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void exportCallback(MlirStringRef message, void *userData) {
  printf("%.*s", (int)message.length, message.data);
}

void exportVerilog(MlirContext ctx, bool disableOptimization) {
  // clang-format off
  const char *testFIR =
    "firrtl.circuit \"ExportTestSimpleModule\" {\n"
    "  firrtl.module @ExportTestSimpleModule(in %in_1: !firrtl.uint<32>,\n"
    "                                        in %in_2: !firrtl.uint<32>,\n"
    "                                        out %out: !firrtl.uint<32>) {\n"
    "    %0 = firrtl.and %in_1, %in_2 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>\n"
    "    %1 = firrtl.and %0, %in_2 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>\n"
    "    firrtl.connect %out, %1 : !firrtl.uint<32>, !firrtl.uint<32>\n"
    "  }\n"
    "}\n";
  // clang-format on
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(testFIR));

  MlirPassManager pm = mlirPassManagerCreate(ctx);

  CirctFirtoolFirtoolOptions options = circtFirtoolOptionsCreateDefault();
  circtFirtoolOptionsSetDisableOptimization(options, disableOptimization);

  MlirLogicalResult result =
      circtFirtoolPopulatePreprocessTransforms(pm, options);
  assert(mlirLogicalResultIsSuccess(result));

  result = circtFirtoolPopulateCHIRRTLToLowFIRRTL(pm, options);
  assert(mlirLogicalResultIsSuccess(result));

  result = circtFirtoolPopulateLowFIRRTLToHW(
      pm, options, mlirStringRefCreateFromCString("-"));
  assert(mlirLogicalResultIsSuccess(result));

  result = circtFirtoolPopulateHWToSV(pm, options);
  assert(mlirLogicalResultIsSuccess(result));

  result = circtFirtoolPopulateExportVerilog(pm, options, exportCallback, NULL);
  assert(mlirLogicalResultIsSuccess(result));

  result = circtFirtoolPopulateFinalizeIR(pm, options);
  assert(mlirLogicalResultIsSuccess(result));

  result = mlirPassManagerRunOnOp(pm, mlirModuleGetOperation(module));
  assert(mlirLogicalResultIsSuccess(result));
}

void testExportVerilog(MlirContext ctx) {
  exportVerilog(ctx, false);

  // CHECK:      module ExportTestSimpleModule(  // -:2:3
  // CHECK-NEXT:   input  [31:0] in_1,   // -:2:44
  // CHECK-NEXT:                 in_2,   // -:3:44
  // CHECK-NEXT:   output [31:0] out     // -:4:45
  // CHECK-NEXT: );
  // CHECK-EMPTY:
  // CHECK-NEXT:   assign out = in_1 & in_2;     // -:2:3, :5:10, :6:10
  // CHECK-NEXT: endmodule

  exportVerilog(ctx, true);

  // CHECK:      module ExportTestSimpleModule(  // -:2:3
  // CHECK-NEXT:   input  [31:0] in_1,   // -:2:44
  // CHECK-NEXT:                 in_2,   // -:3:44
  // CHECK-NEXT:   output [31:0] out     // -:4:45
  // CHECK-NEXT: );
  // CHECK-EMPTY:
  // CHECK-NEXT:   assign out = in_1 & in_2 & in_2;      // -:2:3, :5:10, :6:10
  // CHECK-NEXT: endmodule
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__firrtl__(), ctx);
  testExportVerilog(ctx);
  return 0;
}
