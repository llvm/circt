// RUN: circt-opt %s | circt-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: circt-opt %s --canonicalize | FileCheck %s --check-prefix=CANON
// RUN: circt-opt %s --canonicalize --cse --om-elaborate-object='target-class=Top' | FileCheck %s --check-prefix=ELAB
// RUN: circt-opt %s --om-elaborate-object='all-public-classes allow-unevaluated' --om-elaborate-object='target-class=Top' | FileCheck %s --check-prefix=ELAB
// RUN: circt-opt %s --emit-bytecode | circt-opt | FileCheck %s --check-prefix=ROUNDTRIP

!items = !om.collection<!om.integer>

om.class @Owner() -> (items: !items) {
  %items = om.collection : !items
  om.class.fields %items : !items
}

// Writer has no fields and its result is unused. Its contributions must survive.
// A push into a parameter cannot be folded until the class is inlined.
// CANON-LABEL: om.class @Writer
// CANON: om.collection_push %items, %value
om.class @Writer(%items: !items, %value: !om.integer) {
  om.collection_push %items, %value : !items, !om.integer
  om.class.fields
}

// ROUNDTRIP-LABEL: om.class @Initial
// ROUNDTRIP: om.collection %value, %value {test = "keep"} : !om.collection<!om.integer>
// ROUNDTRIP: om.collection_push
// CANON-LABEL: om.class @Initial
// CANON-NOT: om.collection_push
// CANON: om.collection %value, %value, %value {test = "keep"} : !om.collection<!om.integer>
// CANON-NOT: om.collection_push
// CANON: om.class.fields
om.class @Initial(%value: !om.integer) -> (items: !items, unknown: !items) {
  %items = om.collection %value, %value {test = "keep"} : !items
  om.collection_push %items, %value : !items, !om.integer
  // Contributions to an unknown collection are unobservable.
  %unknown = om.unknown : !items
  om.collection_push %unknown, %value : !items, !om.integer
  om.class.fields %items, %unknown : !items, !items
}

// ROUNDTRIP-LABEL: om.class @Top()
// ROUNDTRIP: om.collection : !om.collection<!om.integer>
// ROUNDTRIP: om.collection_push
// Elaboration folds every contribution into the allocation.
// ELAB-LABEL: om.class @Top()
// ELAB: %[[ONE:.+]] = om.constant #om.integer<1 : si8>
// ELAB-NEXT: om.collection %[[ONE]], %[[ONE]], %[[ONE]] : !om.collection<!om.integer>
// ELAB-NEXT: om.collection : !om.collection<!om.integer>
// ELAB-NOT: om.collection_push
// ELAB-NOT: om.object @Writer
// ELAB: om.class.fields
om.class @Top() -> (items: !items, independent: !items) {
  %owner = om.object @Owner() : () -> !om.class.type<@Owner>
  %items = om.object.field %owner["items"] : (!om.class.type<@Owner>) -> !items
  %independent = om.collection : !items
  %one = om.constant #om.integer<1 : si8> : !om.integer
  %a = om.object @Writer(%items, %one) : (!items, !om.integer) -> !om.class.type<@Writer>
  %b = om.object @Writer(%items, %one) : (!items, !om.integer) -> !om.class.type<@Writer>
  om.collection_push %items, %one : !items, !om.integer
  om.class.fields %items, %independent : !items, !items
}
