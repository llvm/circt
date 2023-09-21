/*===- om.c - Simple test of OM C APIs ------------------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-om-test 2>&1 | FileCheck %s
 */

#include "circt-c/Dialect/OM.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include <mlir-c/Support.h>
#include <stdio.h>

void testEvaluator(MlirContext ctx) {
  const char *testIR =
      "module {"
      "  om.class @Test(%param: !om.integer) {"
      "    om.class.field @field, %param : !om.integer"
      "    %0 = om.object @Child() : () -> !om.class.type<@Child>"
      "    om.class.field @child, %0 : !om.class.type<@Child>"
      "  }"
      "  om.class @Child() {"
      "    %0 = om.constant 14 : i64"
      "    %1 = om.constant 15 : i64"
      "    %2 = om.list_create %0, %1 : i64"
      "    %3 = om.tuple_create %2, %0 : !om.list<i64>, i64"
      "    om.class.field @foo, %0 : i64"
      "    om.class.field @bar, %2 : !om.list<i64>"
      "    om.class.field @baz, %3 : tuple<!om.list<i64>, i64>"
      "  }"
      "}";

  // Set up the Evaluator.
  MlirModule testModule =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(testIR));

  OMEvaluator evaluator = omEvaluatorNew(testModule);

  MlirAttribute className =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("Test"));

  // Test instantiation failure.
  OMEvaluatorValue failedObject =
      omEvaluatorInstantiate(evaluator, className, 0, 0);

  // CHECK: error: actual parameter list length (0) does not match
  // CHECK: object is null: 1
  fprintf(stderr, "object is null: %d\n",
          omEvaluatorObjectIsNull(failedObject));

  // Test instantiation success.

  OMEvaluatorValue actualParam = omEvaluatorValueFromPrimitive(
      omIntegerAttrGet(mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 8), 42)));

  OMEvaluatorValue object =
      omEvaluatorInstantiate(evaluator, className, 1, &actualParam);

  // Test Object type.

  MlirType objectType = omEvaluatorObjectGetType(object);

  // CHECK: !om.class.type<@Test>
  mlirTypeDump(objectType);

  bool isClassType = omTypeIsAClassType(objectType);

  // CHECK: object type is class type: 1
  fprintf(stderr, "object type is class type: %d\n", isClassType);

  // Test get field failure.

  MlirAttribute missingFieldName =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("foo"));

  OMEvaluatorValue missingField =
      omEvaluatorObjectGetField(object, missingFieldName);

  // CHECK: error: field "foo" does not exist
  // CHECK: field is null: 1
  fprintf(stderr, "field is null: %d\n", omEvaluatorValueIsNull(missingField));

  // Test get field success.

  MlirAttribute fieldName =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("field"));

  OMEvaluatorValue field = omEvaluatorObjectGetField(object, fieldName);

  // CHECK: field is object: 0
  fprintf(stderr, "field is object: %d\n", omEvaluatorValueIsAObject(field));
  // CHECK: field is primitive: 1
  fprintf(stderr, "field is primitive: %d\n",
          omEvaluatorValueIsAPrimitive(field));

  MlirAttribute fieldValue = omEvaluatorValueGetPrimitive(field);

  // CHECK: #om.integer<42 : i8> : !om.integer
  mlirAttributeDump(fieldValue);

  // Test get field success for child object.

  MlirAttribute childFieldName =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("child"));

  OMEvaluatorValue child = omEvaluatorObjectGetField(object, childFieldName);

  MlirAttribute fieldNamesO = omEvaluatorObjectGetFieldNames(object);
  // CHECK: ["child", "field"]
  mlirAttributeDump(fieldNamesO);

  // CHECK: 0
  fprintf(stderr, "child object is null: %d\n", omEvaluatorObjectIsNull(child));

  OMEvaluatorValue foo = omEvaluatorObjectGetField(
      child, mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("foo")));

  OMEvaluatorValue bar = omEvaluatorObjectGetField(
      child, mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("bar")));

  OMEvaluatorValue baz = omEvaluatorObjectGetField(
      child, mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("baz")));

  MlirAttribute fieldNamesC = omEvaluatorObjectGetFieldNames(child);

  // CHECK: ["bar", "baz", "foo"]
  mlirAttributeDump(fieldNamesC);

  // CHECK: child object field `foo` is primitive: 1
  fprintf(stderr, "child object field `foo` is primitive: %d\n",
          omEvaluatorValueIsAPrimitive(foo));

  MlirAttribute fooValue = omEvaluatorValueGetPrimitive(foo);

  // CHECK: 14 : i64
  mlirAttributeDump(fooValue);

  // CHECK: child object field `bar` is list: 1
  fprintf(stderr, "child object field `bar` is list: %d\n",
          omEvaluatorValueIsAList(bar));

  // CHECK: 15 : i64
  mlirAttributeDump(
      omEvaluatorValueGetPrimitive(omEvaluatorListGetElement(bar, 1)));

  // CHECK: child object field `baz` is tuple: 1
  fprintf(stderr, "child object field `baz` is tuple: %d\n",
          omEvaluatorValueIsATuple(baz));

  // CHECK: 14 : i64
  mlirAttributeDump(
      omEvaluatorValueGetPrimitive(omEvaluatorTupleGetElement(baz, 1)));
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__om__(), ctx);
  testEvaluator(ctx);
  return 0;
}
