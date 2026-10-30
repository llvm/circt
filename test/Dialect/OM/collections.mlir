// RUN: circt-opt %s | circt-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: circt-opt %s --canonicalize --cse --om-elaborate-object='target-class=Top' | FileCheck %s --check-prefix=ELAB
// RUN: circt-opt %s --om-elaborate-object='all-public-classes allow-unevaluated' --om-elaborate-object='target-class=Top' | FileCheck %s --check-prefix=ELAB
// RUN: circt-opt %s --emit-bytecode | circt-opt | FileCheck %s --check-prefix=ROUNDTRIP

!items = !om.collection<!om.integer>

om.class @Owner() -> (items: !items) {
  %items = om.collection : !items
  om.class.fields %items : !items
}

// Writer has no fields and its result is unused. Its contributions must survive.
om.class @Writer(%items: !items, %value: !om.integer) {
  om.collection_push %items, %value : !items, !om.integer
  om.class.fields
}

// ROUNDTRIP-LABEL: om.class @Top()
// ROUNDTRIP: om.collection : !om.collection<!om.integer>
// ROUNDTRIP: om.collection_push
// ELAB-LABEL: om.class @Top()
// ELAB-COUNT-2: om.collection : !om.collection<!om.integer>
// ELAB-COUNT-3: om.collection_push
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
