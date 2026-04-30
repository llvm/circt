// RUN: circt-opt -om-elaborate-object='target-class=Top' %s | FileCheck %s --check-prefix=TOP
// RUN: circt-opt -om-elaborate-object='test=true' %s | FileCheck %s --check-prefixes=TOP,CHECK

!list = !om.class.type<@LinkedList>

om.class @InputBox(%input: !om.integer) -> (value: !om.integer) {
  om.class.fields %input : !om.integer
}

om.class @LinkedList(%input: !om.integer, %next: !list) -> (value: !om.integer, next: !list) {
  %box = om.object @InputBox(%input) : (!om.integer) -> !om.class.type<@InputBox>
  %value = om.object.field %box["value"] : (!om.class.type<@InputBox>) -> !om.integer
  om.class.fields %value, %next : !om.integer, !list
}

// TOP-LABEL: om.class @Top() -> (list: !om.class.type<@LinkedList>, head: !om.integer, tail: !om.class.type<@LinkedList>) {
// TOP-DAG:   %[[TWO:.+]] = om.constant #om.integer<2 : i6> : !om.integer
// TOP-DAG:   %[[ONE:.+]] = om.constant #om.integer<1 : i6> : !om.integer
// TOP:   %[[tail:.+]] = om.elaborated_object @LinkedList(%[[TWO]], %[[list:.+]]) : (!om.integer, !om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
// TOP:   %[[list]] = om.elaborated_object @LinkedList(%[[ONE]], %[[tail]]) : (!om.integer, !om.class.type<@LinkedList>) -> !om.class.type<@LinkedList>
// TOP:   om.class.fields %[[list]], %[[ONE]], %[[tail]] : !om.class.type<@LinkedList>, !om.integer, !om.class.type<@LinkedList>
// TOP: }
om.class @Top() -> (list: !list, head: !om.integer, tail: !list) {
  %one = om.constant #om.integer<1 : i6> : !om.integer
  %two = om.constant #om.integer<2 : i6> : !om.integer
  %tail = om.object @LinkedList(%two, %list) : (!om.integer, !list) -> !list
  %list = om.object @LinkedList(%one, %tail) : (!om.integer, !list) -> !list
  %head = om.object.field %list["value"] : (!list) -> !om.integer
  om.class.fields %list, %head, %tail : !list, !om.integer, !list
}


// Test nested field references
om.class @Domain(%in: !om.string) -> (out: !om.string) {
  om.class.fields %in : !om.string
}

// CHECK-LABEL: om.class @NestedFieldTop() -> (result: !om.string) {
// CHECK:   %[[STR:.+]] = om.constant "A" : !om.string
// CHECK:   om.class.fields %[[STR]] : !om.string
// CHECK: }
om.class @NestedFieldTop() -> (result: !om.string) {
  %0 = om.constant "A" : !om.string
  %1 = om.object @Domain(%0) : (!om.string) -> !om.class.type<@Domain>
  %2 = om.object.field %1["out"] : (!om.class.type<@Domain>) -> !om.string
  %3 = om.object @Domain(%2) : (!om.string) -> !om.class.type<@Domain>
  %4 = om.object.field %3["out"] : (!om.class.type<@Domain>) -> !om.string
  om.class.fields %4 : !om.string
}


// Test integer arithmetic with object field access
om.class @ValueBox(%val: !om.integer) -> (value: !om.integer) {
  om.class.fields %val : !om.integer
}

// CHECK-LABEL: om.class @IntegerArithTop() -> (result: !om.integer) {
// CHECK:   %[[RESULT:.+]] = om.constant #om.integer<3 : si3> : !om.integer
// CHECK:   om.class.fields %[[RESULT]] : !om.integer
// CHECK: }

om.class @IntegerArithTop() -> (result: !om.integer) {
  %1 = om.constant #om.integer<1 : si3> : !om.integer
  %2 = om.constant #om.integer<2 : si3> : !om.integer
  %box1 = om.object @ValueBox(%1) : (!om.integer) -> !om.class.type<@ValueBox>
  %val1 = om.object.field %box1["value"] : (!om.class.type<@ValueBox>) -> !om.integer
  %box2 = om.object @ValueBox(%2) : (!om.integer) -> !om.class.type<@ValueBox>
  %val2 = om.object.field %box2["value"] : (!om.class.type<@ValueBox>) -> !om.integer
  %sum = om.integer.add %val1, %val2 : !om.integer
  om.class.fields %sum : !om.integer
}

// Test list operations with object fields
om.class @ListBox(%l: !om.list<!om.integer>) -> (value: !om.list<!om.integer>) {
  om.class.fields %l : !om.list<!om.integer>
}

// CHECK-LABEL: om.class @ListConcatTop() -> (result: !om.list<!om.integer>) {
// CHECK-DAG:   %[[ZERO:.+]] = om.constant #om.integer<0 : i8> : !om.integer
// CHECK-DAG:   %[[ONE:.+]] = om.constant #om.integer<1 : i8> : !om.integer
// CHECK-DAG:   %[[TWO:.+]] = om.constant #om.integer<2 : i8> : !om.integer
// CHECK:   %[[L0:.+]] = om.list_create %[[ZERO]], %[[ONE]]
// CHECK:   %[[L2:.+]] = om.list_create %[[TWO]]
// CHECK:   %[[CONCAT:.+]] = om.list_concat %[[L0]], %[[L2]]
// CHECK:   om.class.fields %[[CONCAT]] : !om.list<!om.integer>
// CHECK: }
om.class @ListConcatTop() -> (result: !om.list<!om.integer>) {
  %0 = om.constant #om.integer<0 : i8> : !om.integer
  %1 = om.constant #om.integer<1 : i8> : !om.integer
  %2 = om.constant #om.integer<2 : i8> : !om.integer
  %l0 = om.list_create %0, %1 : !om.integer
  %box = om.object @ListBox(%l0) : (!om.list<!om.integer>) -> !om.class.type<@ListBox>
  %l1 = om.object.field %box["value"] : (!om.class.type<@ListBox>) -> !om.list<!om.integer>
  %l2 = om.list_create %2 : !om.integer
  %concat = om.list_concat %l1, %l2 : !om.list<!om.integer>
  om.class.fields %concat : !om.list<!om.integer>
}

// Test property assertions with true/passing conditions
// CHECK-LABEL: om.class @AssertTrue() {
// CHECK-NOT:   om.property_assert
om.class @AssertTrue() {
  %true = om.constant true
  om.property_assert %true, "this should pass" : i1
  om.class.fields
}

// CHECK-LABEL: om.class @AssertInteger() {
// CHECK-NOT:   om.property_assert
om.class @AssertInteger() {
  %one = om.constant 1 : i1
  om.property_assert %one, "one is true" : i1
  om.class.fields
}

// Test property assertion with unknown condition (should pass and be erased)
// CHECK-LABEL: om.class @AssertUnknown() {
// CHECK-NOT:   om.property_assert
om.class @AssertUnknown() {
  %unknown = om.unknown : i1
  om.property_assert %unknown, "unknown condition" : i1
  om.class.fields
}

// Test external class instantiation (should be replaced with unknown)
om.class.extern @ExternalModule(%param: !om.integer) -> (output: !om.integer) {}

// CHECK-LABEL: om.class @UseExtern() -> (result: !om.integer) {
// CHECK:   %[[UNKNOWN:.+]] = om.unknown : !om.integer
// CHECK:   om.class.fields %[[UNKNOWN]]
// CHECK: }
om.class @UseExtern() -> (result: !om.integer) {
  %input = om.constant #om.integer<42 : si64> : !om.integer
  %ext = om.object @ExternalModule(%input) : (!om.integer) -> !om.class.type<@ExternalModule>
  %result = om.object.field %ext["output"] : (!om.class.type<@ExternalModule>) -> !om.integer
  om.class.fields %result : !om.integer
}

