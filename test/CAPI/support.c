//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: circt-capi-support-test 2>&1 | FileCheck %s
 */

#include "circt-c/Support/InstanceGraph.h"

#include "circt-c/Dialect/AIG.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/Seq.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void testInstancePath(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__hw__(), ctx);
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__seq__(), ctx);

  // clang-format off
  const char *moduleStr =
      "hw.module private @foo(in %c : !seq.clock, in %a: i1, out x: i1) {\n"
      "  %0 = seq.compreg %0, %c : i1\n"
      "  hw.output %0: i1\n"
      "}\n"
      "hw.module private @bar(in %c : !seq.clock, in %a: i1, out x: i1) {\n"
      "  %0 = hw.instance \"foo\" @foo(c: %c: !seq.clock, a: %a: i1) -> (x: i1)\n"
      "  hw.output %0: i1\n"
      "}\n"
      "hw.module private @top(in %c : !seq.clock, in %a: i1, out x: i1) {\n"
      "  %0 = hw.instance \"bar\" @bar(c: %c: !seq.clock, a: %a: i1) -> (x: i1)\n"
      "  hw.output %0: i1\n"
      "}\n";
  // clang-format on

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleStr));

  MlirOperation moduleOp = mlirModuleGetOperation(module);

  // Note: We test IgraphInstancePath functionality through the AIG longest path
  // analysis since it provides access to InstancePath objects. While
  // InstancePath objects are created in InstancePathCache, that interface is
  // not yet exposed through the C API. Using the analysis results is sufficient
  // for testing purposes without needing to expose additional InstancePathCache
  // APIs.

  AIGLongestPathAnalysis analysis =
      aigLongestPathAnalysisCreate(moduleOp, true, false, false);

  MlirStringRef moduleName = mlirStringRefCreateFromCString("top");
  AIGLongestPathCollection collection =
      aigLongestPathAnalysisGetAllPaths(analysis, moduleName, true);

  if (aigLongestPathCollectionIsNull(collection) ||
      aigLongestPathCollectionGetSize(collection) < 1)
    return;

  // Get instance path of fanin object.
  IgraphInstancePath instancePath =
      aigLongestPathObjectGetInstancePath(aigLongestPathDataflowPathGetFanIn(
          aigLongestPathCollectionGetDataflowPath(collection, 0)));

  if (instancePath.size != 2) {
    printf("Instance path size mismatch\n");
    return;
  }
  MlirOperation bar = igraphInstancePathGet(instancePath, 0);
  MlirOperation foo = igraphInstancePathGet(instancePath, 1);

  // CHECK: hw.instance "bar" @bar
  mlirOperationDump(bar);
  // CHECK-NEXT: hw.instance "foo" @foo
  mlirOperationDump(foo);

  aigLongestPathCollectionDestroy(collection);
  aigLongestPathAnalysisDestroy(analysis);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

int main(void) {
  testInstancePath();

  printf("=== All tests completed ===\n");
  // CHECK: === All tests completed ===

  return 0;
}
