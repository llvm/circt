// RUN: circt-opt %s --canonicalize --cse | FileCheck %s
// RUN: circt-opt %s --om-elaborate-object='all-public-classes' | FileCheck %s

// Collect pushes before and after an allocation in block order. Keep initial
// elements first, preserve duplicates and attributes, and do not merge distinct
// allocations with equal contents.
// CHECK-LABEL: om.class @Contributions
// CHECK-NOT: om.collection_push
// CHECK: %[[FIRST:.+]] = om.collection %seed, %second, %first, %second {test = "keep"}
// CHECK-NEXT: %[[OTHER:.+]] = om.collection %seed, %second, %first, %second {test = "keep"}
// CHECK-NEXT: %[[EMPTY:.+]] = om.collection : !om.collection<!om.integer>
// CHECK-NOT: om.collection_push
// CHECK: om.class.fields %[[FIRST]], %[[OTHER]], %[[EMPTY]]
om.class @Contributions(%seed: !om.integer, %first: !om.integer, %second: !om.integer) -> (
  items: !om.collection<!om.integer>,
  other: !om.collection<!om.integer>,
  empty: !om.collection<!om.integer>
) {
  om.collection_push %items, %second : !om.collection<!om.integer>, !om.integer
  %items = om.collection %seed {test = "keep"} : !om.collection<!om.integer>
  %other = om.collection %seed, %second, %first, %second {test = "keep"} : !om.collection<!om.integer>
  %empty = om.collection : !om.collection<!om.integer>
  om.collection_push %items, %first : !om.collection<!om.integer>, !om.integer
  om.collection_push %items, %second : !om.collection<!om.integer>, !om.integer
  om.class.fields %items, %other, %empty : !om.collection<!om.integer>, !om.collection<!om.integer>, !om.collection<!om.integer>
}

// An unknown element does not make membership unknown. Only pushes whose
// destination is unknown are erased.
// CHECK-LABEL: om.class @Unknowns
// CHECK: %[[UNKNOWN:.+]] = om.unknown : !om.integer
// CHECK: %[[ITEMS:.+]] = om.collection %value, %[[UNKNOWN]]
// CHECK: %[[MISSING:.+]] = om.unknown : !om.collection<!om.integer>
// CHECK-NOT: om.collection_push
// CHECK: om.class.fields %[[ITEMS]], %[[MISSING]]
om.class @Unknowns(%value: !om.integer) -> (
  items: !om.collection<!om.integer>,
  missing: !om.collection<!om.integer>
) {
  %unknown = om.unknown : !om.integer
  %items = om.collection %value : !om.collection<!om.integer>
  %missing = om.unknown : !om.collection<!om.integer>
  om.collection_push %items, %unknown : !om.collection<!om.integer>, !om.integer
  om.collection_push %missing, %value : !om.collection<!om.integer>, !om.integer
  om.class.fields %items, %missing : !om.collection<!om.integer>, !om.collection<!om.integer>
}

// A collection used as another push's element is not that push's destination.
// CHECK-LABEL: om.class @Nested
// CHECK: %[[INNER:.+]] = om.collection %value : !om.collection<!om.integer>
// CHECK-NEXT: %[[OUTER:.+]] = om.collection %[[INNER]] : !om.collection<!om.collection<!om.integer>>
// CHECK-NOT: om.collection_push
// CHECK: om.class.fields %[[OUTER]]
om.class @Nested(%value: !om.integer) -> (items: !om.collection<!om.collection<!om.integer>>) {
  %inner = om.collection : !om.collection<!om.integer>
  %outer = om.collection : !om.collection<!om.collection<!om.integer>>
  om.collection_push %inner, %value : !om.collection<!om.integer>, !om.integer
  om.collection_push %outer, %inner : !om.collection<!om.collection<!om.integer>>, !om.collection<!om.integer>
  om.class.fields %outer : !om.collection<!om.collection<!om.integer>>
}

// Preserve graph references, including a collection containing itself via any.
// CHECK-LABEL: om.class @SelfReference
// CHECK: %[[ITEMS:.+]] = om.collection %[[SELF:.+]] : !om.collection<!om.any>
// CHECK-NEXT: %[[SELF]] = om.any_cast %[[ITEMS]]
// CHECK-NOT: om.collection_push
// CHECK: om.class.fields %[[ITEMS]]
om.class @SelfReference() -> (items: !om.collection<!om.any>) {
  %items = om.collection : !om.collection<!om.any>
  %self = om.any_cast %items : (!om.collection<!om.any>) -> !om.any
  om.collection_push %items, %self : !om.collection<!om.any>, !om.any
  om.class.fields %items : !om.collection<!om.any>
}

// Parameter contributions cannot be folded until an allocation is available.
// CHECK-LABEL: om.class @Parameter
// CHECK: om.collection_push %items, %value
om.class @Parameter(%items: !om.collection<!om.integer>, %value: !om.integer) {
  om.collection_push %items, %value : !om.collection<!om.integer>, !om.integer
  om.class.fields
}
