// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects),firrtl-check-comb-loops))' | FileCheck %s

// Mutually referencing object-model nodes are not combinational hardware
// loops. Creating an object reference does not eagerly evaluate its fields.
firrtl.circuit "ObjectGraph" {
  firrtl.class @Node(in %ref: !firrtl.anyref, out %next: !firrtl.anyref) {
    firrtl.propassign %next, %ref : !firrtl.anyref
  }
  // CHECK-LABEL: firrtl.module @ObjectGraph
  firrtl.module @ObjectGraph(out %out: !firrtl.anyref) {
    %a_ref = firrtl.wire : !firrtl.anyref
    %b_ref = firrtl.wire : !firrtl.anyref
    // CHECK: %a = firrtl.object.create(%{{.*}})
    %a = firrtl.object.create(%b_ref) : (!firrtl.anyref) -> !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    // CHECK: %b = firrtl.object.create(%{{.*}})
    %b = firrtl.object.create(%a_ref) : (!firrtl.anyref) -> !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    %a_cast = firrtl.object.anyref_cast %a : !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    %b_cast = firrtl.object.anyref_cast %b : !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    firrtl.propassign %a_ref, %a_cast : !firrtl.anyref
    firrtl.propassign %b_ref, %b_cast : !firrtl.anyref
    firrtl.propassign %out, %a_cast : !firrtl.anyref
  }
}
