//===- aig.c - Test for AIG C API ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: circt-capi-aig-test 2>&1 | FileCheck %s
 */

#include "circt-c/Dialect/AIG.h"
#include "circt-c/Dialect/HW.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void testAIGDialectRegistration(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__aig__(), ctx);

  MlirDialect aig = mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("aig"));
  assert(!mlirDialectIsNull(aig));

  printf("AIG dialect registration: PASS\n");
  // CHECK: AIG dialect registration: PASS

  mlirContextDestroy(ctx);
}

void testLongestPathAnalysisBasic(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__aig__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__hw__(), ctx);

  // Create a simple MLIR module with AIG operations
  const char *moduleStr =
    "hw.module @test_aig(in %a: i1, in %b: i1, out out: i1) {\n"
    "  %0 = aig.and_inv %a, %b : i1\n"
    "  hw.output %0 : i1\n"
    "}\n";

  MlirModule module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleStr));
  assert(!mlirModuleIsNull(module));

