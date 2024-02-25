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

  // CHECK: FIRRTL version 4.0.0
  // CHECK-NEXT: circuit ExportTestSimpleModule :
  // CHECK-NEXT:   module ExportTestSimpleModule : @[- 2:3]
  // CHECK-NEXT:     input in_1 : UInt<32> @[- 2:44]
  // CHECK-NEXT:     input in_2 : UInt<32> @[- 3:44]
  // CHECK-NEXT:     output out : UInt<32> @[- 4:45]
  // CHECK-EMPTY:
  // CHECK-NEXT:     connect out, and(in_1, in_2) @[- 6:5]
}

void testValueFoldFlow(MlirContext ctx) {
  // clang-format off
  const char *testFIR =
    "firrtl.circuit \"ValueFoldFlowTest\" {\n"
    "  firrtl.module @ValueFoldFlowTest(in %in: !firrtl.uint<32>,\n"
    "                                   out %out: !firrtl.uint<32>) {\n"
    "    firrtl.connect %out, %in : !firrtl.uint<32>, !firrtl.uint<32>\n"
    "  }\n"
    "}\n";
  // clang-format on
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(testFIR));
  MlirBlock mlirModule = mlirModuleGetBody(module);
  MlirBlock firCircuit = mlirRegionGetFirstBlock(
      mlirOperationGetRegion(mlirBlockGetFirstOperation(mlirModule), 0));
  MlirBlock firModule = mlirRegionGetFirstBlock(
      mlirOperationGetRegion(mlirBlockGetFirstOperation(firCircuit), 0));

  MlirValue in = mlirBlockGetArgument(firModule, 0);
  MlirValue out = mlirBlockGetArgument(firModule, 1);

  assert(firrtlValueFoldFlow(in, FIRRTL_VALUE_FLOW_SOURCE) ==
         FIRRTL_VALUE_FLOW_SOURCE);
  assert(firrtlValueFoldFlow(out, FIRRTL_VALUE_FLOW_SOURCE) ==
         FIRRTL_VALUE_FLOW_SINK);
}

void testImportAnnotations(MlirContext ctx) {
  // clang-format off
  const char *testFIR =
    "firrtl.circuit \"AnnoTest\" {\n"
    "  firrtl.module @AnnoTest(in %in: !firrtl.uint<32>) {}\n"
    "}\n";
  // clang-format on
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(testFIR));
  MlirBlock mlirModule = mlirModuleGetBody(module);
  MlirOperation firCircuit = mlirBlockGetFirstOperation(mlirModule);

  const char *rawAnnotationsJSON = "[{\
    \"class\":\"firrtl.transforms.DontTouchAnnotation\",\
    \"target\":\"~AnnoTest|AnnoTest>in\"\
  }]";
  MlirAttribute rawAnnotationsAttr;
  bool succeeded = firrtlImportAnnotationsFromJSONRaw(
      ctx, mlirStringRefCreateFromCString(rawAnnotationsJSON),
      &rawAnnotationsAttr);
  assert(succeeded);
  mlirOperationSetAttributeByName(
      firCircuit, mlirStringRefCreateFromCString("rawAnnotations"),
      rawAnnotationsAttr);

  mlirOperationPrint(mlirModuleGetOperation(module), exportCallback, NULL);

  // clang-format off
  // CHECK: firrtl.circuit "AnnoTest" attributes {rawAnnotations = [{class = "firrtl.transforms.DontTouchAnnotation", target = "~AnnoTest|AnnoTest>in"}]} {
  // clang-format on
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__firrtl__(), ctx);
  testExport(ctx);
  testValueFoldFlow(ctx);
  testImportAnnotations(ctx);
  return 0;
}
