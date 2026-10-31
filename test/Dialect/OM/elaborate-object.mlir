// RUN: circt-opt -om-elaborate-object='target-class=Top' %s | FileCheck %s --check-prefix=TOP
// RUN: circt-opt -om-elaborate-object='target-class=UseExtern' %s | FileCheck %s --check-prefix=STRICT-EXTERN
// RUN: circt-opt -om-elaborate-object='all-public-classes=true allow-unevaluated=true' %s | FileCheck %s --check-prefixes=CHECK,PUBLIC,COLLECTION,FOLD
// RUN: circt-opt -om-elaborate-object='target-class=CollectionTop' %s | FileCheck %s --check-prefix=COLLECTION

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

!items = !om.collection<!om.integer>

om.class private @CollectionOwner() -> (items: !items) {
  %items = om.collection : !items
  om.class.fields %items : !items
}

// A parameter contribution remains until inlining exposes the allocation.
// The writer's unused result must not discard its contribution.
om.class private @CollectionWriter(%items: !items, %value: !om.integer) {
  om.collection_push %items, %value : !items, !om.integer
  om.class.fields
}

// COLLECTION-LABEL: om.class @CollectionTop()
// Constant folding can change definition order; sort by the final order.
// COLLECTION: %[[TWO:.+]] = om.constant #om.integer<2 : si8>
// COLLECTION-NEXT: %[[ONE:.+]] = om.constant #om.integer<1 : si8>
// COLLECTION-NEXT: %[[ITEMS:.+]] = om.collection %[[TWO]], %[[TWO]], %[[ONE]] : !om.collection<!om.integer>
// COLLECTION-NEXT: %[[OTHER:.+]] = om.collection : !om.collection<!om.integer>
// COLLECTION-NOT: om.collection_push
// COLLECTION-NOT: om.object @CollectionWriter
// COLLECTION: om.class.fields %[[ITEMS]], %[[OTHER]]
om.class @CollectionTop() -> (items: !items, independent: !items) {
  %owner = om.object @CollectionOwner() : () -> !om.class.type<@CollectionOwner>
  %items = om.object.field %owner["items"] : (!om.class.type<@CollectionOwner>) -> !items
  %other = om.object @CollectionOwner() : () -> !om.class.type<@CollectionOwner>
  %independent = om.object.field %other["items"] : (!om.class.type<@CollectionOwner>) -> !items
  %one = om.constant #om.integer<1 : si8> : !om.integer
  %two = om.constant #om.integer<2 : si8> : !om.integer
  %a = om.object @CollectionWriter(%items, %two) : (!items, !om.integer) -> !om.class.type<@CollectionWriter>
  %b = om.object @CollectionWriter(%items, %one) : (!items, !om.integer) -> !om.class.type<@CollectionWriter>
  om.collection_push %items, %two : !items, !om.integer
  om.class.fields %items, %independent : !items, !items
}

// Fold contributions during inlining, then sort all operands by final IR value
// order. Arguments follow declaration order, irrespective of push visitation.
// Preserve initial elements, duplicates, attributes, and allocation identity.
// FOLD-LABEL: om.class @CollectionContributions
// FOLD-NOT: om.collection_push
// FOLD: %[[FIRST:.+]] = om.collection %seed, %first, %second, %second {test = "keep"}
// FOLD-NEXT: %[[OTHER:.+]] = om.collection %seed, %first, %second, %second {test = "keep"}
// FOLD-NEXT: %[[EMPTY:.+]] = om.collection : !om.collection<!om.integer>
// FOLD-NOT: om.collection_push
// FOLD: om.class.fields %[[FIRST]], %[[OTHER]], %[[EMPTY]]
om.class @CollectionContributions(%seed: !om.integer, %first: !om.integer, %second: !om.integer) -> (
  items: !om.collection<!om.integer>,
  other: !om.collection<!om.integer>,
  empty: !om.collection<!om.integer>
) {
  om.collection_push %items, %second : !om.collection<!om.integer>, !om.integer
  %items = om.collection %seed {test = "keep"} : !om.collection<!om.integer>
  %other = om.collection %seed, %second, %first, %second {test = "keep"} : !om.collection<!om.integer>
  %empty = om.collection : !om.collection<!om.integer>
  %writer = om.object @CollectionWriter(%items, %first) : (!om.collection<!om.integer>, !om.integer) -> !om.class.type<@CollectionWriter>
  om.collection_push %items, %second : !om.collection<!om.integer>, !om.integer
  om.class.fields %items, %other, %empty : !om.collection<!om.integer>, !om.collection<!om.integer>, !om.collection<!om.integer>
}

// Unknown elements remain contributions; unknown destinations are discarded.
// FOLD-LABEL: om.class @CollectionUnknowns
// FOLD: %[[UNKNOWN:.+]] = om.unknown : !om.integer
// FOLD: %[[ITEMS:.+]] = om.collection %value, %[[UNKNOWN]]
// FOLD: %[[MISSING:.+]] = om.unknown : !om.collection<!om.integer>
// FOLD-NOT: om.collection_push
// FOLD: om.class.fields %[[ITEMS]], %[[MISSING]]
om.class @CollectionUnknowns(%value: !om.integer) -> (
  items: !om.collection<!om.integer>, missing: !om.collection<!om.integer>
) {
  %unknown = om.unknown : !om.integer
  %items = om.collection %value : !om.collection<!om.integer>
  %missing = om.unknown : !om.collection<!om.integer>
  om.collection_push %items, %unknown : !om.collection<!om.integer>, !om.integer
  om.collection_push %missing, %value : !om.collection<!om.integer>, !om.integer
  om.class.fields %items, %missing : !om.collection<!om.integer>, !om.collection<!om.integer>
}

// A collection used as an element is not that push's destination.
// FOLD-LABEL: om.class @CollectionNested
// FOLD: %[[INNER:.+]] = om.collection %value : !om.collection<!om.integer>
// FOLD-NEXT: %[[OUTER:.+]] = om.collection %[[INNER]] : !om.collection<!om.collection<!om.integer>>
// FOLD-NOT: om.collection_push
// FOLD: om.class.fields %[[OUTER]]
om.class @CollectionNested(%value: !om.integer) -> (items: !om.collection<!om.collection<!om.integer>>) {
  %inner = om.collection : !om.collection<!om.integer>
  %outer = om.collection : !om.collection<!om.collection<!om.integer>>
  om.collection_push %inner, %value : !om.collection<!om.integer>, !om.integer
  om.collection_push %outer, %inner : !om.collection<!om.collection<!om.integer>>, !om.collection<!om.integer>
  om.class.fields %outer : !om.collection<!om.collection<!om.integer>>
}

// Remove unobservable allocations after folding their contributions.
// FOLD-LABEL: om.class @CollectionUnused
// FOLD-NOT: om.collection
// FOLD-NOT: om.constant
// FOLD: om.class.fields
om.class @CollectionUnused() {
  %empty = om.collection : !om.collection<!om.integer>
  %items = om.collection : !om.collection<!om.integer>
  %one = om.constant #om.integer<1 : si8> : !om.integer
  om.collection_push %items, %one : !om.collection<!om.integer>, !om.integer
  om.class.fields
}
