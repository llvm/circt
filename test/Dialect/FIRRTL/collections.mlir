// RUN: circt-opt %s | circt-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: circt-opt %s --firrtl-lower-classes | FileCheck %s --check-prefix=LOWER

firrtl.circuit "Collections" {
  // ROUNDTRIP-LABEL: firrtl.module @Collections
  // ROUNDTRIP: firrtl.prop.collection : !firrtl.mutable_collection<integer>
  // ROUNDTRIP-COUNT-3: firrtl.prop.push
  // LOWER-LABEL: om.class @Collections_Class(%basepath: !om.basepath, %items: !om.collection<!om.integer>)
  // LOWER-SAME: -> (result: !om.collection<!om.integer>)
  // LOWER: %[[LOCAL:.*]] = om.collection : !om.collection<!om.integer>
  // LOWER-COUNT-3: om.collection_push
  // LOWER: om.class.fields %items : !om.collection<!om.integer>
  firrtl.module @Collections(in %items: !firrtl.mutable_collection<integer>, out %result: !firrtl.mutable_collection<integer>) {
    %local = firrtl.prop.collection : !firrtl.mutable_collection<integer>
    %one = firrtl.integer 1
    firrtl.prop.push %items, %one : !firrtl.mutable_collection<integer>, !firrtl.integer
    firrtl.prop.push %items, %one : !firrtl.mutable_collection<integer>, !firrtl.integer
    firrtl.prop.push %local, %one : !firrtl.mutable_collection<integer>, !firrtl.integer
    firrtl.propassign %result, %items : !firrtl.mutable_collection<integer>
  }

  // No property ports: effect-only property usage must still generate a class.
  // LOWER-LABEL: om.class private @EffectOnly_Class
  // LOWER: om.collection : !om.collection<!om.integer>
  // LOWER: om.collection_push
  firrtl.module private @EffectOnly() {
    %items = firrtl.prop.collection : !firrtl.mutable_collection<integer>
    %one = firrtl.integer 1
    firrtl.prop.push %items, %one : !firrtl.mutable_collection<integer>, !firrtl.integer
  }

  // LOWER-LABEL: om.class private @Nested
  // LOWER-SAME: !om.collection<!om.list<!om.collection<i1>>>
  // LOWER: om.collection : !om.collection<!om.list<!om.collection<i1>>>
  firrtl.class private @Nested(out %result: !firrtl.mutable_collection<list<mutable_collection<bool>>>) {
    %items = firrtl.prop.collection : !firrtl.mutable_collection<list<mutable_collection<bool>>>
    firrtl.propassign %result, %items : !firrtl.mutable_collection<list<mutable_collection<bool>>>
  }
}
