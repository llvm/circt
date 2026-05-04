// RUN: circt-opt -om-elaborate-object='target-class=Top' %s | FileCheck %s --check-prefix=TOP
// RUN: circt-opt -om-elaborate-object='test=true' %s | FileCheck %s --check-prefixes=TOP,CHECK

// CHECK-LABEL: om.class @StringOps() -> (str: !om.string, concat: !om.string) {
// CHECK-DAG:   %[[HELLO:.+]] = om.constant "hello" : !om.string
// CHECK-DAG:   %[[CONCAT:.+]] = om.constant "hello world" : !om.string
// CHECK:   om.class.fields %[[HELLO]], %[[CONCAT]] : !om.string, !om.string
// CHECK: }
om.class @StringOps() -> (str: !om.string, concat: !om.string) {
  %hello = om.constant "hello" : !om.string
  %world = om.constant " world" : !om.string
  %concat = om.string.concat %hello, %world : !om.string
  om.class.fields %hello, %concat : !om.string, !om.string
}

// Test list creation.
// Note that this is currently not folded because list_create doesn't have a folder.
// Even when it had a folder we cannot fully evaluate list_concat since objects cannot be
// representable with attributes.
// CHECK-LABEL: om.class @SimpleList() -> (result: !om.list<!om.integer>) {
// CHECK-DAG:   %[[C1:.+]] = om.constant #om.integer<1 : si8> : !om.integer
// CHECK-DAG:   %[[C2:.+]] = om.constant #om.integer<2 : si8> : !om.integer
// CHECK:   %[[LIST:.+]] = om.list_create %[[C1]], %[[C2]]
// CHECK:   om.class.fields %[[LIST]] : !om.list<!om.integer>
// CHECK: }
om.class @SimpleList() -> (result: !om.list<!om.integer>) {
  %c1 = om.constant #om.integer<1 : si8> : !om.integer
  %c2 = om.constant #om.integer<2 : si8> : !om.integer
  %list = om.list_create %c1, %c2 : !om.integer
  om.class.fields %list : !om.list<!om.integer>
}

// Test integer operations (add, mul, shr, shl)
// CHECK-LABEL: om.class @IntegerOps() -> (sum: !om.integer, product: !om.integer, shr: !om.integer, shl: !om.integer) {
// CHECK-DAG:   %[[SUM:.+]] = om.constant #om.integer<7 : si4> : !om.integer
// CHECK-DAG:   %[[PRODUCT:.+]] = om.constant #om.integer<12 : si8> : !om.integer
// CHECK-DAG:   %[[SHR:.+]] = om.constant #om.integer<2 : si8> : !om.integer
// CHECK-DAG:   %[[SHL:.+]] = om.constant #om.integer<16 : si8> : !om.integer
// CHECK:   om.class.fields %[[SUM]], %[[PRODUCT]], %[[SHR]], %[[SHL]] : !om.integer, !om.integer, !om.integer, !om.integer
// CHECK: }
om.class @IntegerOps() -> (sum: !om.integer, product: !om.integer, shr: !om.integer, shl: !om.integer) {
  %c3 = om.constant #om.integer<3 : si4> : !om.integer
  %c4 = om.constant #om.integer<4 : si4> : !om.integer
  %sum = om.integer.add %c3, %c4 : !om.integer
  %c3_8 = om.constant #om.integer<3 : si8> : !om.integer
  %c4_8 = om.constant #om.integer<4 : si8> : !om.integer
  %product = om.integer.mul %c3_8, %c4_8 : !om.integer
  %c8 = om.constant #om.integer<8 : si8> : !om.integer
  %c2 = om.constant #om.integer<2 : si8> : !om.integer
  %shr_result = om.integer.shr %c8, %c2 : !om.integer
  %shl_result = om.integer.shl %c4_8, %c2 : !om.integer
  om.class.fields %sum, %product, %shr_result, %shl_result : !om.integer, !om.integer, !om.integer, !om.integer
}

// More complex tests

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


// CHECK-LABEL: om.class @NestedFieldTop() -> (result: !om.integer) {
// CHECK:   %[[V:.+]] = om.constant #om.integer<1 : i6>
// CHECK:   om.class.fields %[[V]] : !om.integer
// CHECK: }
om.class @NestedFieldTop() -> (result: !om.integer) {
  %0 = om.constant #om.integer<1 : i6> : !om.integer
  %1 = om.object @InputBox(%0) : (!om.integer) -> !om.class.type<@InputBox>
  %2 = om.object.field %1["value"] : (!om.class.type<@InputBox>) -> !om.integer
  %3 = om.object @InputBox(%2) : (!om.integer) -> !om.class.type<@InputBox>
  %4 = om.object.field %3["value"] : (!om.class.type<@InputBox>) -> !om.integer
  om.class.fields %4 : !om.integer
}

// CHECK-LABEL: om.class @IntegerArithTop() -> (result: !om.integer) {
// CHECK:   %[[RESULT:.+]] = om.constant #om.integer<3 : si3> : !om.integer
// CHECK:   om.class.fields %[[RESULT]] : !om.integer
// CHECK: }

om.class @IntegerArithTop() -> (result: !om.integer) {
  %1 = om.constant #om.integer<1 : si3> : !om.integer
  %2 = om.constant #om.integer<2 : si3> : !om.integer
  %box1 = om.object @InputBox(%1) : (!om.integer) -> !om.class.type<@InputBox>
  %val1 = om.object.field %box1["value"] : (!om.class.type<@InputBox>) -> !om.integer
  %box2 = om.object @InputBox(%2) : (!om.integer) -> !om.class.type<@InputBox>
  %val2 = om.object.field %box2["value"] : (!om.class.type<@InputBox>) -> !om.integer
  %sum = om.integer.add %val1, %val2 : !om.integer
  om.class.fields %sum : !om.integer
}

// Test property assertion with true/passing condition
// CHECK-LABEL: om.class @AssertTrue() {
// CHECK-NOT:   om.property_assert
om.class @AssertTrue() {
  %true = om.constant true
  om.property_assert %true, "this should pass" : i1
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

