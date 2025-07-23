//===----------------------------------------------------------------------===//
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
      "hw.module private @ch(in %c : !seq.clock, in %a: i2, out x: i2) {\n"
      "  %p = seq.compreg %q, %c : i2\n"
      "  %q = aig.and_inv %p, %p {sv.namehint = \"q\"}: i2\n"
      "  hw.output %p: i2\n"
      "}\n"
      "hw.module private @top(in %c : !seq.clock, in %a: i2) {\n"
      "  %0 = hw.instance \"i2\" @ch(c: %c: !seq.clock, a: %a: i2) -> (x: i2)\n"
      "  %1 = hw.instance \"i2\" @ch(c: %c: !seq.clock, a: %a: i2) -> (x: i2)\n"
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
    // CHECK: Path count with elaboration: 4

    pathCount = aigLongestPathCollectionGetSize(collection2);
    printf("Path count without elaboration: %zu\n", pathCount);
    // CHECK: Path count without elaboration: 2

    // Test DataflowPath API
    if (pathCount > 0) {
      AIGLongestPathDataflowPath path =
          aigLongestPathCollectionGetDataflowPath(collection1, 0);

      int64_t delay = aigLongestPathDataflowPathGetDelay(path);
      printf("Path delay: %lld\n", (long long)delay);
      // CHECK: Path delay: 1

      AIGLongestPathObject fanIn = aigLongestPathDataflowPathGetFanIn(path);
      AIGLongestPathObject fanOut = aigLongestPathDataflowPathGetFanOut(path);

      // Test Object API
      MlirStringRef fanInName = aigLongestPathObjectName(fanIn);
      MlirStringRef fanOutName = aigLongestPathObjectName(fanOut);
      size_t fanInBitPos = aigLongestPathObjectBitPos(fanIn);
      size_t fanOutBitPos = aigLongestPathObjectBitPos(fanOut);

      printf("FanIn: %.*s[%zu]\n", (int)fanInName.length, fanInName.data,
             fanInBitPos);
      printf("FanOut: %.*s[%zu]\n", (int)fanOutName.length, fanOutName.data,
             fanOutBitPos);
      // CHECK: FanIn: p[[[BIT:[0-9]]]]
      // CHECK: FanOut: p[[[BIT]]]

      // Test instance path
      IgraphInstancePath fanInPath = aigLongestPathObjectGetInstancePath(fanIn);
      IgraphInstancePath fanOutPath =
          aigLongestPathObjectGetInstancePath(fanOut);
      printf("FanIn instance path size: %zu\n", fanInPath.size);
      printf("FanOut instance path size: %zu\n", fanOutPath.size);
      // CHECK: FanIn instance path size: 1
      // CHECK: FanOut instance path size: 1

      // Test History API
      AIGLongestPathHistory history =
          aigLongestPathDataflowPathGetHistory(path);
      bool isEmpty = aigLongestPathHistoryIsEmpty(history);
      printf("History is empty: %s\n", isEmpty ? "true" : "false");
      // CHECK: History is empty: false

      if (!isEmpty) {
        AIGLongestPathObject historyObject;
        int64_t historyDelay;
        MlirStringRef historyComment;

        aigLongestPathHistoryGetHead(history, &historyObject, &historyDelay,
                                     &historyComment);
        printf("History head delay: %lld\n", (long long)historyDelay);
        printf("History head comment: %.*s\n", (int)historyComment.length,
               historyComment.data);
        // CHECK: History head delay: 1
        // CHECK: History head comment: namehint

        AIGLongestPathHistory tail = aigLongestPathHistoryGetTail(history);
        bool tailIsEmpty = aigLongestPathHistoryIsEmpty(tail);
        printf("History tail is empty: %s\n", tailIsEmpty ? "true" : "false");
        // CHECK: History tail is empty: true
      }

      // Test root operation
      MlirOperation root = aigLongestPathDataflowPathGetRoot(path);
      printf("Root operation is null: %s\n",
             mlirOperationIsNull(root) ? "true" : "false");
      // CHECK: Root operation is null: false
    }

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

    if (!aigLongestPathCollectionIsNull(collection)) {
      size_t size = aigLongestPathCollectionGetSize(collection);
      if (size > 1) {
        AIGLongestPathDataflowPath path =
            aigLongestPathCollectionGetDataflowPath(collection, 1);

        AIGLongestPathHistory history =
            aigLongestPathDataflowPathGetHistory(path);
        bool isEmpty = aigLongestPathHistoryIsEmpty(history);
        printf("History without debug points is empty: %s\n",
               isEmpty ? "true" : "false");
        // CHECK: History without debug points is empty: true
      }
    }

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

  // Cleanup
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
