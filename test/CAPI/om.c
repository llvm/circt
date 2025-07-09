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

void testTypes(MlirContext ctx) {
  // Test StringType creation and type checking
  MlirType stringType = omStringTypeGet(ctx);

  // CHECK: string type is StringType: 1
  fprintf(stderr, "string type is StringType: %d\n",
          omTypeIsAStringType(stringType));

  // CHECK: !om.string
  mlirTypeDump(stringType);

  MlirTypeID stringTypeID = omStringTypeGetTypeID();
  MlirTypeID actualStringTypeID = mlirTypeGetTypeID(stringType);

  // CHECK: StringType TypeID matches: 1
  fprintf(stderr, "StringType TypeID matches: %d\n",
          mlirTypeIDEqual(stringTypeID, actualStringTypeID));
}

void testListAttr(MlirContext ctx) {
  // Test creating ListAttr with integer elements
  MlirType i64Type = mlirIntegerTypeGet(ctx, 64);

  MlirAttribute elem1 = mlirIntegerAttrGet(i64Type, 42);
  MlirAttribute elem2 = mlirIntegerAttrGet(i64Type, 84);

  const MlirAttribute elements[] = {elem1, elem2};
  MlirAttribute listAttr = omListAttrGet(i64Type, 2, elements);

  // CHECK: list attr is ListAttr: 1
  fprintf(stderr, "list attr is ListAttr: %d\n", omAttrIsAListAttr(listAttr));

  // CHECK: list attr num elements: 2
  fprintf(stderr, "list attr num elements: %ld\n",
          omListAttrGetNumElements(listAttr));

  // CHECK: #om.list<i64, [42, 84]>
  mlirAttributeDump(listAttr);

  // Test accessing elements
  MlirAttribute retrievedElem1 = omListAttrGetElement(listAttr, 0);
  MlirAttribute retrievedElem2 = omListAttrGetElement(listAttr, 1);

  // CHECK: first element: 42 : i64
  fprintf(stderr, "first element: ");
  mlirAttributeDump(retrievedElem1);

  // CHECK: second element: 84 : i64
  fprintf(stderr, "second element: ");
  mlirAttributeDump(retrievedElem2);

  // Test creating empty ListAttr
  MlirAttribute emptyListAttr = omListAttrGet(i64Type, 0, NULL);

  // CHECK: empty list attr is ListAttr: 1
  fprintf(stderr, "empty list attr is ListAttr: %d\n",
          omAttrIsAListAttr(emptyListAttr));

  // CHECK: empty list attr num elements: 0
  fprintf(stderr, "empty list attr num elements: %ld\n",
          omListAttrGetNumElements(emptyListAttr));

  // Test creating ListAttr with string elements
  MlirType stringType = omStringTypeGet(ctx);

  MlirAttribute strElem = mlirStringAttrTypedGet(
      stringType, mlirStringRefCreateFromCString("hello"));
  const MlirAttribute stringElements[] = {strElem};
  MlirAttribute stringListAttr = omListAttrGet(stringType, 1, stringElements);

  // CHECK: string list attr is ListAttr: 1
  fprintf(stderr, "string list attr is ListAttr: %d\n",
          omAttrIsAListAttr(stringListAttr));

  // CHECK: #om.list<!om.string, ["hello" : !om.string]>
  mlirAttributeDump(stringListAttr);
}

void testEvaluator(MlirContext ctx) {
  const char *testIR =
      "module {"
      "  om.class @Test(%param: !om.integer) -> (field: !om.integer, child: "
      "!om.class.type<@Child>) {"
      "    %0 = om.object @Child() : () -> !om.class.type<@Child>"
      "    om.class.fields %param, %0 : !om.integer, !om.class.type<@Child>"
      "  }"
      "  om.class @Child() -> (foo: i64, bar: !om.list<i64>){"
      "    %0 = om.constant 14 : i64"
      "    %1 = om.constant 15 : i64"
      "    %2 = om.list_create %0, %1 : i64"
      "    om.class.fields %0, %2 : i64, !om.list<i64>"
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

  MlirAttribute fieldNamesC = omEvaluatorObjectGetFieldNames(child);

  // CHECK: ["bar", "foo"]
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
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__om__(), ctx);
  MlirDialectHandle omDialectHandle = mlirGetDialectHandle__om__();

  // Load the OM dialect to ensure types are properly initialized
  mlirContextGetOrLoadDialect(ctx,
                              mlirDialectHandleGetNamespace(omDialectHandle));

  testTypes(ctx);
  testListAttr(ctx);
  testEvaluator(ctx);
  return 0;
}
