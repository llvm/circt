// RUN: circt-opt -om-elaborate-object='target-class=Top' %s | FileCheck %s --check-prefix=TOP
// RUN: circt-opt -om-elaborate-object='target-class=UseExtern' %s | FileCheck %s --check-prefix=STRICT-EXTERN
// RUN: circt-opt -om-elaborate-object='all-public-classes=true allow-unevaluated=true' %s | FileCheck %s --check-prefixes=CHECK,PUBLIC

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

// Constant primitive lists are folded to attributes; lists of objects cannot
// be represented as attributes and therefore remain list_create operations.
// CHECK-LABEL: om.class @SimpleList() -> (result: !om.list<!om.integer>) {
// CHECK:   %[[LIST:.+]] = om.constant #om.list<!om.integer, [#om.integer<1 : si8> : !om.integer, #om.integer<2 : si8> : !om.integer]> : !om.list<!om.integer>
// CHECK:   om.class.fields %[[LIST]] : !om.list<!om.integer>
// CHECK: }
om.class @SimpleList() -> (result: !om.list<!om.integer>) {
  %c1 = om.constant #om.integer<1 : si8> : !om.integer
  %c2 = om.constant #om.integer<2 : si8> : !om.integer
  %list = om.list_create %c1, %c2 : !om.integer
  om.class.fields %list : !om.list<!om.integer>
}

// Test direct list membership folding across list construction, concatenation,
// and constant list attributes.
// CHECK-LABEL: om.class @ContainsTests
// CHECK:   %[[TRUE:.+]] = om.constant true
// CHECK-NEXT:   %[[FALSE:.+]] = om.constant false
// CHECK-NEXT:   om.class.fields %[[TRUE]], %[[TRUE]], %[[FALSE]], %[[FALSE]], %[[TRUE]], %[[TRUE]], %[[TRUE]], %[[FALSE]], %[[TRUE]], %[[FALSE]], %[[TRUE]] : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
// CHECK: }
om.class @ContainsTests(%identity: !om.string) -> (
    strStart: i1, strMiddle: i1, strAbsent: i1, emptyResult: i1,
    concatEnd: i1, constantResult: i1, intFound: i1, intAbsent: i1,
    boolFound: i1, boolAbsent: i1, identityResult: i1) {
  %sA = om.constant "a" : !om.string
  %sB = om.constant "b" : !om.string
  %sAbsent = om.constant "absent" : !om.string
  %strings = om.list_create %sA, %sB, %sB : !om.string
  %strStart = om.list_contains %strings, %sA : !om.list<!om.string>
  %strMiddle = om.list_contains %strings, %sB : !om.list<!om.string>
  %strAbsent = om.list_contains %strings, %sAbsent : !om.list<!om.string>
  %empty = om.list_create : !om.string
  %emptyResult = om.list_contains %empty, %sA : !om.list<!om.string>
  %head = om.list_create %sA : !om.string
  %tail = om.list_create %sB : !om.string
  %joined = om.list_concat %head, %tail : !om.list<!om.string>
  %concatEnd = om.list_contains %joined, %sB : !om.list<!om.string>
  %bigA = om.constant #om.integer<36755551979133953793 : si66> : !om.integer
  %bigB = om.constant #om.integer<36755551979133953793 : si67> : !om.integer
  %bigOther = om.constant #om.integer<36755551979133953794 : si67> : !om.integer
  %integers = om.list_create %bigA : !om.integer
  %intFound = om.list_contains %integers, %bigB : !om.list<!om.integer>
  %listConstant = om.constant #om.list<!om.integer, [#om.integer<36755551979133953793 : si66>]> : !om.list<!om.integer>
  %constantResult = om.list_contains %listConstant, %bigB : !om.list<!om.integer>
  %intAbsent = om.list_contains %integers, %bigOther : !om.list<!om.integer>

  %true = om.constant true
  %false = om.constant false
  %bools = om.list_create %true : i1
  %boolFound = om.list_contains %bools, %true : !om.list<i1>
  %boolAbsent = om.list_contains %bools, %false : !om.list<i1>
  %identityList = om.list_create %identity : !om.string
  %identityResult = om.list_contains %identityList, %identity :
      !om.list<!om.string>
  om.class.fields %strStart, %strMiddle, %strAbsent, %emptyResult, %concatEnd,
      %constantResult, %intFound, %intAbsent, %boolFound, %boolAbsent,
      %identityResult : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// Class arguments become constants through object inlining before the folder
// retries membership.
om.class @ContainsParameter(%values: !om.list<!om.integer>,
                            %element: !om.integer) -> (found: i1) {
  %found = om.list_contains %values, %element : !om.list<!om.integer>
  om.class.fields %found : i1
}

// CHECK-LABEL: om.class @ContainsParameterTop() -> (found: i1) {
// CHECK:   %[[TRUE:.+]] = om.constant true
// CHECK-NEXT:   om.class.fields %[[TRUE]] : i1
// CHECK: }
om.class @ContainsParameterTop() -> (found: i1) {
  %one = om.constant #om.integer<1 : si8> : !om.integer
  %two = om.constant #om.integer<2 : si8> : !om.integer
  %values = om.list_create %one, %two : !om.integer
  %child = om.object @ContainsParameter(%values, %two) :
      (!om.list<!om.integer>, !om.integer) -> !om.class.type<@ContainsParameter>
  %found = om.object.field %child["found"] :
      (!om.class.type<@ContainsParameter>) -> i1
  om.class.fields %found : i1
}

// Unknown inputs use the existing pure-op propagation policy.
// CHECK-LABEL: om.class @ContainsUnknown() -> (elementResult: i1, listResult: i1) {
// CHECK:   %0 = om.unknown : !om.string
// CHECK-NEXT:   %1 = om.unknown : i1
// CHECK-NEXT:   %2 = om.unknown : !om.list<!om.string>
// CHECK-NEXT:   %3 = om.unknown : i1
// CHECK-NEXT:   om.class.fields %1, %3 : i1, i1
// CHECK: }
om.class @ContainsUnknown() -> (elementResult: i1, listResult: i1) {
  %unknownElement = om.unknown : !om.string
  %list = om.list_create : !om.string
  %unknownElementResult = om.list_contains %list, %unknownElement :
      !om.list<!om.string>
  %unknownList = om.unknown : !om.list<!om.string>
  %knownElement = om.constant "value" : !om.string
  %unknownListResult = om.list_contains %unknownList, %knownElement :
      !om.list<!om.string>
  om.class.fields %unknownElementResult, %unknownListResult : i1, i1
}

// Same-SSA membership matches fold even when the directly constructed list
// also contains an explicit unknown value; the nonmatching result propagates.
// Keep the input unknown distinct from the propagated result unknown.
// CHECK-LABEL: om.class @ContainsUnknownMembers() -> (foo: i1, bar: i1, baz: i1) {
// CHECK:   %[[TRUE:.+]] = om.constant true
// CHECK-NEXT:   %1 = om.unknown : i1
// CHECK-NEXT:   %2 = om.unknown : !om.list<i1>
// CHECK-NEXT:   %3 = om.unknown : i1
// CHECK-NEXT:   om.class.fields %[[TRUE]], %3, %[[TRUE]] : i1, i1, i1
// CHECK: }
om.class @ContainsUnknownMembers() -> (foo: i1, bar: i1, baz: i1) {
  %0 = om.unknown : i1
  %1 = om.constant false
  %2 = om.constant true
  %3 = om.list_create %0, %1 : i1
  %4 = om.list_contains %3, %1 : !om.list<i1>
  %5 = om.list_contains %3, %2 : !om.list<i1>
  %6 = om.list_create %1, %0 : i1
  %7 = om.list_contains %6, %1 : !om.list<i1>
  om.class.fields %4, %5, %7 : i1, i1, i1
}

// Test integer operations (add, mul, shr, shl)
// CHECK-LABEL: om.class @IntegerOps() -> (sum: !om.integer, product: !om.integer, shr: !om.integer, shl: !om.integer) {
// CHECK-DAG:   %[[SUM:.+]] = om.constant #om.integer<7 : si4> : !om.integer
// CHECK-DAG:   %[[PRODUCT:.+]] = om.constant #om.integer<12 : si8> : !om.integer
// CHECK-DAG:   %[[SHR:.+]] = om.constant #om.integer<2 : si8> : !om.integer
// CHECK-DAG:   %[[SHL:.+]] = om.constant #om.integer<16 : si10> : !om.integer
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

// Test shl with shift amount exceeding the lhs bitwidth (extends lhs to lhsWidth + shiftAmt)
// CHECK-LABEL: om.class @IntegerShlAutoExtend() -> (result: !om.integer) {
// CHECK:   %[[R:.+]] = om.constant #om.integer<65536 : si24> : !om.integer
// CHECK:   om.class.fields %[[R]] : !om.integer
// CHECK: }
om.class @IntegerShlAutoExtend() -> (result: !om.integer) {
  %c1 = om.constant #om.integer<1 : si8> : !om.integer
  %c16 = om.constant #om.integer<16 : si8> : !om.integer
  %result = om.integer.shl %c1, %c16 : !om.integer
  om.class.fields %result : !om.integer
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
  %message = om.constant "this should pass" : !om.string
  om.property_assert %true, %message : i1
  om.class.fields
}

// Test property assertion with unknown condition (should pass and be erased)
// CHECK-LABEL: om.class @AssertUnknown() {
// CHECK-NOT:   om.property_assert
om.class @AssertUnknown() {
  %unknown = om.unknown : i1
  %message = om.constant "unknown condition" : !om.string
  om.property_assert %unknown, %message : i1
  om.class.fields
}

// Test external class instantiation (should be replaced with unknown)
om.class.extern @ExternalModule(%param: !om.integer) -> (output: !om.integer) {}

// STRICT-EXTERN-LABEL: om.class @UseExtern() -> (result: !om.integer) {
// STRICT-EXTERN:   %[[UNKNOWN:.+]] = om.unknown : !om.integer
// STRICT-EXTERN:   om.class.fields %[[UNKNOWN]]
// STRICT-EXTERN: }
// PUBLIC-LABEL: om.class @UseExtern() -> (result: !om.integer) {
// PUBLIC:   %[[INPUT:.+]] = om.constant #om.integer<42 : si64> : !om.integer
// PUBLIC:   %[[EXT:.+]] = om.object @ExternalModule(%[[INPUT]]) : (!om.integer) -> !om.class.type<@ExternalModule>
// PUBLIC:   %[[RESULT:.+]] = om.object.field %[[EXT]]["output"] : (!om.class.type<@ExternalModule>) -> !om.integer
// PUBLIC:   om.class.fields %[[RESULT]]
// PUBLIC: }
om.class @UseExtern() -> (result: !om.integer) {
  %input = om.constant #om.integer<42 : si64> : !om.integer
  %ext = om.object @ExternalModule(%input) : (!om.integer) -> !om.class.type<@ExternalModule>
  %result = om.object.field %ext["output"] : (!om.class.type<@ExternalModule>) -> !om.integer
  om.class.fields %result : !om.integer
}

// Test all-public-classes mode elaborates public classes through private
// helpers, but does not elaborate private top-level classes.
// PUBLIC-LABEL: om.class private @PublicModeHelper() -> (value: !om.integer) {
om.class private @PublicModeHelper() -> (value: !om.integer) {
  %value = om.constant #om.integer<7 : si4> : !om.integer
  om.class.fields %value : !om.integer
}

// PUBLIC-LABEL: om.class @PublicModeTop() -> (value: !om.integer) {
// PUBLIC-NEXT:    %[[value:.+]] = om.constant
// PUBLIC-NEXT:   om.class.fields %[[value]]
om.class @PublicModeTop() -> (value: !om.integer) {
  %helper = om.object @PublicModeHelper() : () -> !om.class.type<@PublicModeHelper>
  %value = om.object.field %helper["value"] : (!om.class.type<@PublicModeHelper>) -> !om.integer
  om.class.fields %value : !om.integer
}

// PUBLIC-LABEL: om.class private @PrivateModeTop() -> (value: !om.integer) {
// PUBLIC-NEXT:  om.object @PublicModeHelper()
// PUBLIC-NEXT:  om.object.field
om.class private @PrivateModeTop() -> (value: !om.integer) {
  %helper = om.object @PublicModeHelper() : () -> !om.class.type<@PublicModeHelper>
  %value = om.object.field %helper["value"] : (!om.class.type<@PublicModeHelper>) -> !om.integer
  om.class.fields %value : !om.integer
}
