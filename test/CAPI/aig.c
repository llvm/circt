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
#include "circt-c/Dialect/Seq.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void testAIGDialectRegistration(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__aig__(), ctx);

  MlirDialect aig =
      mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("aig"));

  // CHECK: AIG dialect registration: PASS
  if (!mlirDialectIsNull(aig))
    printf("AIG dialect registration: PASS\n");

  mlirContextDestroy(ctx);
}

void testLongestPathAnalysis(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__aig__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__hw__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__seq__(), ctx);

  // clang-format off
  const char *moduleStr =
      "hw.module private @ch(in %c : !seq.clock, in %a: i1, out x: i1) {\n"
      "  %p = seq.compreg %q, %c : i1\n"
      "  %q = aig.and_inv %p, %p {sv.namehint = \"q\"}: i1\n"
      "  hw.output %p: i1\n"
      "}\n"
      "hw.module private @top(in %c : !seq.clock, in %a: i1) {\n"
      "  %0 = hw.instance \"i1\" @ch(c: %c: !seq.clock, a: %a: i1) -> (x: i1)\n"
      "  %1 = hw.instance \"i2\" @ch(c: %c: !seq.clock, a: %a: i1) -> (x: i1)\n"
      "}\n";
  // clang-format on

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleStr));

  MlirOperation moduleOp = mlirModuleGetOperation(module);

  // Test with debug points enabled
  {
    AIGLongestPathAnalysis analysis =
        aigLongestPathAnalysisCreate(moduleOp, true);

    MlirStringRef moduleName = mlirStringRefCreateFromCString("top");
    AIGLongestPathCollection collection1 =
        aigLongestPathAnalysisGetAllPaths(analysis, moduleName, true);
    AIGLongestPathCollection collection2 =
        aigLongestPathAnalysisGetAllPaths(analysis, moduleName, false);

    size_t pathCount = aigLongestPathCollectionGetSize(collection1);
    printf("Path count with elaboration: %zu\n", pathCount);
    // CHECK: Path count with elaboration: 2

    pathCount = aigLongestPathCollectionGetSize(collection2);
    printf("Path count without elaboration: %zu\n", pathCount);
    // CHECK: Path count without elaboration: 1

    // Test getting individual paths

    // Test multiple path access
    MlirStringRef pathJson = aigLongestPathCollectionGetPath(collection1, 1);
    printf("Path: %.*s\n", (int)pathJson.length, pathJson.data);
    // CHECK: Path: {"fan_out":{"bit_pos":
    // CHECK-SAME: "history":[{

    // Cleanup
    aigLongestPathCollectionDestroy(collection1);
    aigLongestPathCollectionDestroy(collection2);
    aigLongestPathAnalysisDestroy(analysis);

    printf("LongestPathAnalysis with debug points: PASS\n");
    // CHECK: LongestPathAnalysis with debug points: PASS
  }

  // Test without debug points
  {
    AIGLongestPathAnalysis analysis =
        aigLongestPathAnalysisCreate(moduleOp, false);

    MlirStringRef moduleName = mlirStringRefCreateFromCString("top");
    AIGLongestPathCollection collection =
        aigLongestPathAnalysisGetAllPaths(analysis, moduleName, true);

    MlirStringRef pathJson = aigLongestPathCollectionGetPath(collection, 1);
    printf("Path: %.*s\n", (int)pathJson.length, pathJson.data);
    // CHECK: "history":[]

    // Cleanup
    aigLongestPathCollectionDestroy(collection);
    aigLongestPathAnalysisDestroy(analysis);

    printf("LongestPathAnalysis without debug points: PASS\n");
    // CHECK: LongestPathAnalysis without debug points: PASS
  }

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

void testErrorHandling(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__aig__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__hw__(), ctx);

  const char *moduleStr =
      "hw.module @test(in %a: i1, in %b: i1, out out: i1) {\n"
      "  hw.output %a : i1\n"
      "}\n";

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleStr));

  MlirOperation moduleOp = mlirModuleGetOperation(module);

  AIGLongestPathAnalysis analysis =
      aigLongestPathAnalysisCreate(moduleOp, true);

  MlirStringRef invalidModuleName = mlirStringRefCreateFromCString("unknown");
  AIGLongestPathCollection invalidCollection =
      aigLongestPathAnalysisGetAllPaths(analysis, invalidModuleName, true);

  if (aigLongestPathCollectionIsNull(invalidCollection)) {
    // CHECK: Invalid module name handling: PASS
    printf("Invalid module name handling: PASS\n");
  }

  MlirStringRef moduleName = mlirStringRefCreateFromCString("test");
  AIGLongestPathCollection collection =
      aigLongestPathAnalysisGetAllPaths(analysis, moduleName, true);
  MlirStringRef invalidPath = aigLongestPathCollectionGetPath(collection, 100);
  if (invalidPath.length == 0) {
    printf("Invalid path index handling: PASS\n");
    // CHECK: Invalid path index handling: PASS
  }

  // Cleanup
  aigLongestPathCollectionDestroy(collection);
  aigLongestPathAnalysisDestroy(analysis);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

int main(void) {
  testAIGDialectRegistration();
  testLongestPathAnalysis();
  testErrorHandling();

  printf("=== All tests completed ===\n");
  // CHECK: === All tests completed ===

  return 0;
}
