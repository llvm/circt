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
#include <string.h>

#define ARRAY_SIZE(arr) (sizeof((arr)) / sizeof((arr)[0]))

void dumpCallback(MlirStringRef message, void *userData) {
  fprintf(stderr, "%.*s", (int)message.length, message.data);
}

void appendBufferCallback(MlirStringRef message, void *userData) {
  char *buffer = (char *)userData;
  sprintf(buffer + strlen(buffer), "%.*s", (int)message.length, message.data);
}

bool bundleFieldEqual(const FIRRTLBundleField *lhs,
                      const FIRRTLBundleField *rhs) {
  return mlirIdentifierEqual(lhs->name, rhs->name) &&
         lhs->isFlip == rhs->isFlip && mlirTypeEqual(lhs->type, rhs->type);
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

  MlirLogicalResult result = mlirExportFIRRTL(module, dumpCallback, NULL);
  (void)result;
  assert(mlirLogicalResultIsSuccess(result));

  // CHECK: FIRRTL version
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

  assert(firrtlValueFoldFlow(mlirBlockGetArgument(firModule, 0),
                             FIRRTL_VALUE_FLOW_SOURCE) ==
         FIRRTL_VALUE_FLOW_SOURCE);
  assert(firrtlValueFoldFlow(mlirBlockGetArgument(firModule, 1),
                             FIRRTL_VALUE_FLOW_SOURCE) ==
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
  (void)succeeded;
  assert(succeeded);
  mlirOperationSetAttributeByName(
      firCircuit, mlirStringRefCreateFromCString("rawAnnotations"),
      rawAnnotationsAttr);

  mlirOperationPrint(mlirModuleGetOperation(module), dumpCallback, NULL);

  // clang-format off
  // CHECK: firrtl.circuit "AnnoTest" attributes {rawAnnotations = [{class = "firrtl.transforms.DontTouchAnnotation", target = "~AnnoTest|AnnoTest>in"}]} {
  // clang-format on
}

void assertAttrEqual(MlirAttribute lhs, MlirAttribute rhs) {
  char lhsBuffer[256] = {0}, rhsBuffer[256] = {0};
  mlirAttributePrint(lhs, appendBufferCallback, lhsBuffer);
  mlirAttributePrint(rhs, appendBufferCallback, rhsBuffer);
  assert(strcmp(lhsBuffer, rhsBuffer) == 0);
}

void testAttrGetIntegerFromString(MlirContext ctx) {
  // large negative hex
  assertAttrEqual(
      mlirAttributeParseGet(
          ctx, mlirStringRefCreateFromCString("0xFF0000000000000000 : i72")),
      firrtlAttrGetIntegerFromString(
          mlirIntegerTypeGet(ctx, 72), 72,
          mlirStringRefCreateFromCString("FF0000000000000000"), 16));

  // large positive hex
  assertAttrEqual(
      mlirAttributeParseGet(
          ctx, mlirStringRefCreateFromCString("0xFF0000000000000000 : i73")),
      firrtlAttrGetIntegerFromString(
          mlirIntegerTypeGet(ctx, 73), 73,
          mlirStringRefCreateFromCString("FF0000000000000000"), 16));

  // large negative dec
  assertAttrEqual(
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString(
                                     "-12345678912345678912345 : i75")),
      firrtlAttrGetIntegerFromString(
          mlirIntegerTypeGet(ctx, 75), 75,
          mlirStringRefCreateFromCString("-12345678912345678912345"), 10));

  // large positive dec
  assertAttrEqual(
      mlirAttributeParseGet(
          ctx, mlirStringRefCreateFromCString("12345678912345678912345 : i75")),
      firrtlAttrGetIntegerFromString(
          mlirIntegerTypeGet(ctx, 75), 75,
          mlirStringRefCreateFromCString("12345678912345678912345"), 10));

  // small negative hex
  assertAttrEqual(
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("0xFF : i8")),
      firrtlAttrGetIntegerFromString(mlirIntegerTypeGet(ctx, 8), 8,
                                     mlirStringRefCreateFromCString("FF"), 16));

  // small positive hex
  assertAttrEqual(
      mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString("0xFF : i9")),
      firrtlAttrGetIntegerFromString(mlirIntegerTypeGet(ctx, 9), 9,
                                     mlirStringRefCreateFromCString("FF"), 16));

  // small negative dec
  assertAttrEqual(mlirAttributeParseGet(
                      ctx, mlirStringRefCreateFromCString("-114514 : i18")),
                  firrtlAttrGetIntegerFromString(
                      mlirIntegerTypeGet(ctx, 18), 18,
                      mlirStringRefCreateFromCString("-114514"), 10));

  // small positive dec
  assertAttrEqual(mlirAttributeParseGet(
                      ctx, mlirStringRefCreateFromCString("114514 : i18")),
                  firrtlAttrGetIntegerFromString(
                      mlirIntegerTypeGet(ctx, 18), 18,
                      mlirStringRefCreateFromCString("114514"), 10));
}

void testTypeDiscriminantsAndQueries(MlirContext ctx) {
  FIRRTLBundleField bundleFields[] = {
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f1")),
          .isFlip = false,
          .type = firrtlTypeGetUInt(ctx, 32),
      },
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f2")),
          .isFlip = false,
          .type = firrtlTypeGetSInt(ctx, 64),
      },
  };
  FIRRTLBundleField openBundleFields[] = {
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f1")),
          .isFlip = false,
          .type = firrtlTypeGetUInt(ctx, 32),
      },
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f2")),
          .isFlip = false,
          .type = firrtlTypeGetInteger(ctx),
      },
  };
  MlirType bundle =
      firrtlTypeGetBundle(ctx, ARRAY_SIZE(bundleFields), bundleFields);
  bundleFields[1].isFlip = true;
  MlirType bundleContainsFlip =
      firrtlTypeGetBundle(ctx, ARRAY_SIZE(bundleFields), bundleFields);
  MlirType openBundle =
      firrtlTypeGetBundle(ctx, ARRAY_SIZE(openBundleFields), openBundleFields);

  FIRRTLClassElement classElements[] = {
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f1")),
          .type = firrtlTypeGetUInt(ctx, 32),
          .direction = FIRRTL_DIRECTION_IN,
      },
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f2")),
          .type = firrtlTypeGetInteger(ctx),
          .direction = FIRRTL_DIRECTION_OUT,
      },
  };
  MlirType cls = firrtlTypeGetClass(
      ctx, mlirFlatSymbolRefAttrGet(ctx, mlirStringRefCreateFromCString("cls")),
      ARRAY_SIZE(classElements), classElements);

  assert(firrtlTypeIsConst(
      firrtlTypeGetConstType(firrtlTypeGetUInt(ctx, 32), true)));
  assert(firrtlTypeIsConst(firrtlTypeGetConstType(
      firrtlTypeGetConstType(firrtlTypeGetUInt(ctx, 32), false), true)));
  assert(firrtlTypeIsAUInt(firrtlTypeGetUInt(ctx, 32)));
  assert(firrtlTypeIsASInt(firrtlTypeGetSInt(ctx, 32)));
  assert(firrtlTypeIsAClock(firrtlTypeGetClock(ctx)));
  assert(firrtlTypeIsAReset(firrtlTypeGetReset(ctx)));
  assert(firrtlTypeIsAAsyncReset(firrtlTypeGetAsyncReset(ctx)));
  assert(firrtlTypeIsAAnalog(firrtlTypeGetAnalog(ctx, 32)));
  assert(firrtlTypeIsAVector(
      firrtlTypeGetVector(ctx, firrtlTypeGetUInt(ctx, 32), 4)));
  assert(firrtlTypeIsABundle(bundle));
  assert(firrtlTypeIsAOpenBundle(openBundle));
  assert(firrtlTypeIsARef(firrtlTypeGetRef(firrtlTypeGetClock(ctx), false)));
  assert(firrtlTypeIsAAnyRef(firrtlTypeGetAnyRef(ctx)));
  assert(firrtlTypeIsAInteger(firrtlTypeGetInteger(ctx)));
  assert(firrtlTypeIsADouble(firrtlTypeGetDouble(ctx)));
  assert(firrtlTypeIsAString(firrtlTypeGetString(ctx)));
  assert(firrtlTypeIsABoolean(firrtlTypeGetBoolean(ctx)));
  assert(firrtlTypeIsAPath(firrtlTypeGetPath(ctx)));
  assert(firrtlTypeIsAList(firrtlTypeGetList(ctx, firrtlTypeGetInteger(ctx))));
  assert(firrtlTypeIsAClass(cls));
  assert(!firrtlTypeIsConst(firrtlTypeGetUInt(ctx, 32)));
  assert(!firrtlTypeIsConst(
      firrtlTypeGetConstType(firrtlTypeGetUInt(ctx, 32), false)));
  assert(!firrtlTypeIsConst(firrtlTypeGetConstType(
      firrtlTypeGetConstType(firrtlTypeGetUInt(ctx, 32), true), false)));
  assert(!firrtlTypeIsAUInt(firrtlTypeGetSInt(ctx, 32)));
  assert(!firrtlTypeIsASInt(firrtlTypeGetUInt(ctx, 32)));
  assert(!firrtlTypeIsAClock(firrtlTypeGetReset(ctx)));
  assert(!firrtlTypeIsAReset(firrtlTypeGetClock(ctx)));
  assert(!firrtlTypeIsAAsyncReset(firrtlTypeGetReset(ctx)));
  assert(!firrtlTypeIsAAnalog(firrtlTypeGetReset(ctx)));
  assert(!firrtlTypeIsABundle(openBundle));
  assert(!firrtlTypeIsAOpenBundle(bundle));
  assert(!firrtlTypeIsARef(firrtlTypeGetAnyRef(ctx)));
  assert(
      !firrtlTypeIsAAnyRef(firrtlTypeGetRef(firrtlTypeGetClock(ctx), false)));
  assert(!firrtlTypeIsAInteger(firrtlTypeGetDouble(ctx)));
  assert(!firrtlTypeIsADouble(firrtlTypeGetString(ctx)));
  assert(!firrtlTypeIsAString(firrtlTypeGetBoolean(ctx)));
  assert(!firrtlTypeIsABoolean(firrtlTypeGetPath(ctx)));
  assert(!firrtlTypeIsAPath(firrtlTypeGetBoolean(ctx)));
  assert(!firrtlTypeIsAList(cls));
  assert(!firrtlTypeIsAClass(firrtlTypeGetPath(ctx)));

  assert(firrtlTypeGetBitWidth(firrtlTypeGetUInt(ctx, 32), false) == 32);
  assert(firrtlTypeGetBitWidth(firrtlTypeGetSInt(ctx, 32), false) == 32);
  assert(firrtlTypeGetBitWidth(firrtlTypeGetClock(ctx), false) == 1);
  assert(firrtlTypeGetBitWidth(firrtlTypeGetReset(ctx), false) == 1);
  assert(firrtlTypeGetBitWidth(bundle, false) == 96);
  assert(firrtlTypeGetBitWidth(bundleContainsFlip, false) == -1);
  assert(firrtlTypeGetBitWidth(bundleContainsFlip, true) == 96);
  assert(mlirTypeEqual(firrtlTypeGetVectorElement(firrtlTypeGetVector(
                           ctx, firrtlTypeGetUInt(ctx, 32), 4)),
                       firrtlTypeGetUInt(ctx, 32)));
  assert(firrtlTypeGetVectorNumElements(
             firrtlTypeGetVector(ctx, firrtlTypeGetUInt(ctx, 32), 4)) == 4);
  assert(firrtlTypeGetBundleNumFields(bundle) == 2);
  assert(firrtlTypeGetBundleNumFields(openBundle) == 2);
  FIRRTLBundleField field;
  assert(firrtlTypeGetBundleFieldByIndex(bundle, 0, &field) &&
         bundleFieldEqual(&bundleFields[0], &field));
  assert(firrtlTypeGetBundleFieldByIndex(openBundle, 1, &field) &&
         bundleFieldEqual(&openBundleFields[1], &field));
  assert(firrtlTypeGetBundleFieldIndex(
             bundle, mlirStringRefCreateFromCString("f1")) == 0);
  assert(firrtlTypeGetBundleFieldIndex(
             bundle, mlirStringRefCreateFromCString("f2")) == 1);
  assert(firrtlTypeGetBundleFieldIndex(
             openBundle, mlirStringRefCreateFromCString("f1")) == 0);
  assert(firrtlTypeGetBundleFieldIndex(
             openBundle, mlirStringRefCreateFromCString("f2")) == 1);
}

void testTypeGetMaskType(MlirContext ctx) {
  assert(mlirTypeEqual(firrtlTypeGetMaskType(firrtlTypeGetUInt(ctx, 32)),
                       firrtlTypeGetUInt(ctx, 1)));
  assert(mlirTypeEqual(firrtlTypeGetMaskType(firrtlTypeGetSInt(ctx, 64)),
                       firrtlTypeGetUInt(ctx, 1)));

  FIRRTLBundleField lhsFields[] = {
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f1")),
          .isFlip = false,
          .type = firrtlTypeGetUInt(ctx, 32),
      },
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f2")),
          .isFlip = false,
          .type = firrtlTypeGetSInt(ctx, 64),
      },
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f3")),
          .isFlip = false,
          .type = firrtlTypeGetClock(ctx),
      },
  };
  FIRRTLBundleField rhsFields[] = {
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f1")),
          .isFlip = false,
          .type = firrtlTypeGetUInt(ctx, 1),
      },
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f2")),
          .isFlip = false,
          .type = firrtlTypeGetUInt(ctx, 1),
      },
      {
          .name = mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("f3")),
          .isFlip = false,
          .type = firrtlTypeGetUInt(ctx, 1),
      },
  };
  assert(mlirTypeEqual(
      firrtlTypeGetMaskType(
          firrtlTypeGetBundle(ctx, ARRAY_SIZE(lhsFields), lhsFields)),
      firrtlTypeGetBundle(ctx, ARRAY_SIZE(rhsFields), rhsFields)));
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleLoadDialect(mlirGetDialectHandle__firrtl__(), ctx);
  testExport(ctx);
  testValueFoldFlow(ctx);
  testImportAnnotations(ctx);
  testAttrGetIntegerFromString(ctx);
  testTypeDiscriminantsAndQueries(ctx);
  testTypeGetMaskType(ctx);
  return 0;
}
