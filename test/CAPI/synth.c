//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: circt-capi-synth-test 2>&1 | FileCheck %s
 */

#include "circt-c/Dialect/Synth.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/Seq.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void testSynthDialectRegistration(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__synth__(), ctx);

  MlirDialect synth =
      mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("synth"));

  // CHECK: Synth dialect registration: PASS
  if (!mlirDialectIsNull(synth))
    printf("Synth dialect registration: PASS\n");

  mlirContextDestroy(ctx);
}

void testLongestPathAnalysis(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__synth__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__hw__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__seq__(), ctx);

  // clang-format off
  const char *moduleStr =
      "hw.module private @ch(in %c : !seq.clock, in %a: i2, out x: i2) {\n"
      "  %p = seq.compreg %q, %c : i2\n"
      "  %q = synth.aig.and_inv %p, %p {sv.namehint = \"q\"}: i2\n"
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
    SynthLongestPathAnalysis analysis =
        synthLongestPathAnalysisCreate(moduleOp, true, false, false);

    MlirStringRef moduleName = mlirStringRefCreateFromCString("top");
    SynthLongestPathCollection collection1 =
        synthLongestPathAnalysisGetAllPaths(analysis, moduleName, true);
    SynthLongestPathCollection collection2 =
        synthLongestPathAnalysisGetAllPaths(analysis, moduleName, false);

    size_t pathCount = synthLongestPathCollectionGetSize(collection1);
    printf("Path count with elaboration: %zu\n", pathCount);
    // CHECK: Path count with elaboration: 4

    pathCount = synthLongestPathCollectionGetSize(collection2);
    printf("Path count without elaboration: %zu\n", pathCount);
    // CHECK: Path count without elaboration: 2

    // Test DataflowPath API
    if (pathCount > 0) {
      SynthLongestPathDataflowPath path =
          synthLongestPathCollectionGetDataflowPath(collection1, 0);

      int64_t delay = synthLongestPathDataflowPathGetDelay(path);
      printf("Path delay: %lld\n", (long long)delay);
      // CHECK: Path delay: 1

      SynthLongestPathObject fanIn = synthLongestPathDataflowPathGetFanIn(path);
      SynthLongestPathObject fanOut =
          synthLongestPathDataflowPathGetFanOut(path);

      // Test Object API
      MlirStringRef fanInName = synthLongestPathObjectName(fanIn);
      MlirStringRef fanOutName = synthLongestPathObjectName(fanOut);
      size_t fanInBitPos = synthLongestPathObjectBitPos(fanIn);
      size_t fanOutBitPos = synthLongestPathObjectBitPos(fanOut);

      printf("FanIn: %.*s[%zu]\n", (int)fanInName.length, fanInName.data,
             fanInBitPos);
      printf("FanOut: %.*s[%zu]\n", (int)fanOutName.length, fanOutName.data,
             fanOutBitPos);
      // CHECK: FanIn: p[[[BIT:[0-9]]]]
      // CHECK: FanOut: p[[[BIT]]]

      // Test instance path
      IgraphInstancePath fanInPath =
          synthLongestPathObjectGetInstancePath(fanIn);
      IgraphInstancePath fanOutPath =
          synthLongestPathObjectGetInstancePath(fanOut);
      printf("FanIn instance path size: %zu\n", fanInPath.size);
      printf("FanOut instance path size: %zu\n", fanOutPath.size);
      // CHECK: FanIn instance path size: 1
      // CHECK: FanOut instance path size: 1

      // Test History API
      SynthLongestPathHistory history =
          synthLongestPathDataflowPathGetHistory(path);
      bool isEmpty = synthLongestPathHistoryIsEmpty(history);
      printf("History is empty: %s\n", isEmpty ? "true" : "false");
      // CHECK: History is empty: false

      if (!isEmpty) {
        SynthLongestPathObject historyObject;
        int64_t historyDelay;
        MlirStringRef historyComment;

        synthLongestPathHistoryGetHead(history, &historyObject, &historyDelay,
                                       &historyComment);
        printf("History head delay: %lld\n", (long long)historyDelay);
        printf("History head comment: %.*s\n", (int)historyComment.length,
               historyComment.data);
        // CHECK: History head delay: 1
        // CHECK: History head comment: namehint

        SynthLongestPathHistory tail = synthLongestPathHistoryGetTail(history);
        bool tailIsEmpty = synthLongestPathHistoryIsEmpty(tail);
        printf("History tail is empty: %s\n", tailIsEmpty ? "true" : "false");
        // CHECK: History tail is empty: true
      }

      // Test root operation
      MlirOperation root = synthLongestPathDataflowPathGetRoot(path);
      printf("Root operation is null: %s\n",
             mlirOperationIsNull(root) ? "true" : "false");
      // CHECK: Root operation is null: false
    }

    // Cleanup
    synthLongestPathCollectionDestroy(collection1);
    synthLongestPathCollectionDestroy(collection2);
    synthLongestPathAnalysisDestroy(analysis);

    printf("LongestPathAnalysis with debug points: PASS\n");
    // CHECK: LongestPathAnalysis with debug points: PASS
  }

  // Test without debug points
  {
    SynthLongestPathAnalysis analysis =
        synthLongestPathAnalysisCreate(moduleOp, false, false, false);

    MlirStringRef moduleName = mlirStringRefCreateFromCString("top");
    SynthLongestPathCollection collection =
        synthLongestPathAnalysisGetAllPaths(analysis, moduleName, true);

    if (!synthLongestPathCollectionIsNull(collection)) {
      size_t size = synthLongestPathCollectionGetSize(collection);
      if (size > 1) {
        SynthLongestPathDataflowPath path =
            synthLongestPathCollectionGetDataflowPath(collection, 1);

        SynthLongestPathHistory history =
            synthLongestPathDataflowPathGetHistory(path);
        bool isEmpty = synthLongestPathHistoryIsEmpty(history);
        printf("History without debug points is empty: %s\n",
               isEmpty ? "true" : "false");
        // CHECK: History without debug points is empty: true
      }
    }

    // Cleanup
    synthLongestPathCollectionDestroy(collection);
    synthLongestPathAnalysisDestroy(analysis);

    printf("LongestPathAnalysis without debug points: PASS\n");
    // CHECK: LongestPathAnalysis without debug points: PASS
  }

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

void testErrorHandling(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__synth__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__hw__(), ctx);

  const char *moduleStr =
      "hw.module @test(in %a: i1, in %b: i1, out out: i1) {\n"
      "  hw.output %a : i1\n"
      "}\n";

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleStr));

  MlirOperation moduleOp = mlirModuleGetOperation(module);

  SynthLongestPathAnalysis analysis =
      synthLongestPathAnalysisCreate(moduleOp, true, false, false);

  MlirStringRef invalidModuleName = mlirStringRefCreateFromCString("unknown");
  SynthLongestPathCollection invalidCollection =
      synthLongestPathAnalysisGetAllPaths(analysis, invalidModuleName, true);

  if (synthLongestPathCollectionIsNull(invalidCollection)) {
    // CHECK: Invalid module name handling: PASS
    printf("Invalid module name handling: PASS\n");
  }

  // Cleanup
  synthLongestPathAnalysisDestroy(analysis);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

void testGetPathsAndMerge(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__synth__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__hw__(), ctx);

  // clang-format off
  const char *moduleStr =
      "hw.module @m(in %a : i1, in %b: i1, out x: i1) {\n"
      "  %q = synth.aig.and_inv %a, %b : i1\n"
      "  %r = synth.aig.and_inv %q, %b : i1\n"
      "  hw.output %r: i1\n"
      "}\n";
  // clang-format on

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleStr));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  // Create analysis (enable only-max-delay to keep collections small and
  // stable).
  SynthLongestPathAnalysis analysis =
      synthLongestPathAnalysisCreate(moduleOp, /*collectDebugInfo=*/false,
                                     /*keepOnlyMaxDelayPaths=*/true,
                                     /*lazyComputation=*/false);

  // Navigate to @m body and grab the two synth.aig.and_inv results.
  MlirRegion topRegion0 = mlirOperationGetRegion(moduleOp, 0);
  MlirBlock topBlock0 = mlirRegionGetFirstBlock(topRegion0);
  MlirOperation hwModule = mlirBlockGetFirstOperation(topBlock0);

  MlirRegion mRegion = mlirOperationGetRegion(hwModule, 0);
  MlirBlock mBlock = mlirRegionGetFirstBlock(mRegion);
  MlirOperation and1 = mlirBlockGetFirstOperation(mBlock);
  MlirOperation and2 = mlirOperationGetNextInBlock(and1);

  MlirValue val1 = mlirOperationGetResult(and1, 0);
  MlirValue val2 = mlirOperationGetResult(and2, 0);

  SynthLongestPathCollection col1 =
      synthLongestPathAnalysisGetPaths(analysis, val1, /*bitPos=*/0,
                                       /*elaboratePaths=*/false);
  SynthLongestPathCollection col2 =
      synthLongestPathAnalysisGetPaths(analysis, val2, /*bitPos=*/0,
                                       /*elaboratePaths=*/false);

  bool ok = !synthLongestPathCollectionIsNull(col1) &&
            !synthLongestPathCollectionIsNull(col2) &&
            (synthLongestPathCollectionGetSize(col1) > 0) &&
            (synthLongestPathCollectionGetSize(col2) > 0);
  printf("C API get_paths non-empty: %s\n", ok ? "PASS" : "FAIL");
  // CHECK: C API get_paths non-empty: PASS

  size_t s1 = synthLongestPathCollectionGetSize(col1);
  size_t s2 = synthLongestPathCollectionGetSize(col2);
  synthLongestPathCollectionMerge(col1, col2);
  size_t sMerged = synthLongestPathCollectionGetSize(col1);
  bool mergedOk = sMerged == s1 + s2;
  printf("C API collection merge: %s (s1=%zu, s2=%zu, merged=%zu)\n",
         mergedOk ? "PASS" : "FAIL", s1, s2, sMerged);
  // CHECK: C API collection merge: PASS

  synthLongestPathCollectionDestroy(col1);
  synthLongestPathCollectionDestroy(col2);
  synthLongestPathAnalysisDestroy(analysis);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

int main(void) {
  testSynthDialectRegistration();
  testLongestPathAnalysis();
  testErrorHandling();
  testGetPathsAndMerge();

  printf("=== All tests completed ===\n");
  // CHECK: === All tests completed ===

  return 0;
}
