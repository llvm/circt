/*===- firrtl.c - Simple test of FIRRTL C APIs ----------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-firrtl-test 2>&1 | FileCheck %s
 */

#include "circt-c/Dialect/FIRRTL.h"
#include "circt-c/ExportFIRRTL.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void exportCallback(MlirStringRef message, void *userData) {
  printf("%.*s", (int)message.length, message.data);
}

void testExport(MlirContext ctx) {
  // clang-format off
  const char *testFIR =
    "firrtl.circuit \"ExportTestSimpleModule\" {\n"
    "  firrtl.module @ExportTestSimpleModule(in %in_1: !firrtl.uint<32>,\n"
    "                                        in %in_2: !firrtl.uint<32>,\n"
    "                                        out %out: !firrtl.uint<32>) {\n"
    "    %0 = firrtl.and %in_1, %in_2 : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>\n"
    "    firrtl.connect %out, %0 : !firrtl.uint<32>, !firrtl.uint<32>\n"
    "  }\n"
    "}\n";
  // clang-format on
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(testFIR));

  MlirLogicalResult result = mlirExportFIRRTL(module, exportCallback, NULL);
  assert(mlirLogicalResultIsSuccess(result));

  // CHECK: FIRRTL version 3.3.0
  // CHECK-NEXT: circuit ExportTestSimpleModule :
  // CHECK-NEXT:   module ExportTestSimpleModule : @[- 2:3]
  // CHECK-NEXT:     input in_1 : UInt<32> @[- 2:44]
  // CHECK-NEXT:     input in_2 : UInt<32> @[- 3:44]
  // CHECK-NEXT:     output out : UInt<32> @[- 4:45]
  // CHECK-EMPTY:
  // CHECK-NEXT:     connect out, and(in_1, in_2) @[- 6:5]
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__firrtl__(), ctx);
  testExport(ctx);
  return 0;
}
