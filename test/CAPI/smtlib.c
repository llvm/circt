
/*===- smtlib.c - Simple test of SMTLIB C API -----------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-smtlib-test 2>&1 | FileCheck %s
 */

#include "circt-c/Dialect/SMT.h"
#include "circt-c/ExportSMTLIB.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include <assert.h>
#include <stdio.h>

void dumpCallback(MlirStringRef message, void *userData) {
  fprintf(stderr, "%.*s", (int)message.length, message.data);
}

void testExportSMTLIB(MlirContext ctx) {
  // clang-format off
  const char *testSMT = 
    "func.func @test() {\n"
    "  smt.solver() : () -> () { }\n"
    "  return\n"
    "}\n";
  // clang-format on

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(testSMT));

  MlirLogicalResult result = mlirExportSMTLIB(module, dumpCallback, NULL);
  (void)result;
  assert(mlirLogicalResultIsSuccess(result));

  // CHECK: ; solver scope 0
  // CHECK-NEXT: (reset)
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__smt__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__func__(), ctx);
  testExportSMTLIB(ctx);
  return 0;
}