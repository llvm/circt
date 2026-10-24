// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects,firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects),firrtl-check-comb-loops))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects),firrtl-lower-classes))' | FileCheck %s --check-prefix=OM

firrtl.circuit "Wires" {
  firrtl.module @Wires() {}
  firrtl.class @Mixed(in %first: !firrtl.string, out %result: !firrtl.string, in %constant: !firrtl.integer, in %last: !firrtl.integer) {
    firrtl.propassign %result, %first : !firrtl.string
  }
  firrtl.extclass private @Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)
  firrtl.extmodule @External(in arg: !firrtl.integer)

  // Module objects do not need temporary wires.
  // Preserve port order, output accesses, and object attributes. Remove all
  // input assignments and accesses, including unused duplicate accesses.
  // CHECK-LABEL: firrtl.module @MixedInputs
  // OM-LABEL: om.class @MixedInputs_Class
  firrtl.module @MixedInputs(in %arg: !firrtl.string, in %integer: !firrtl.integer, out %out: !firrtl.string) {
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 42
    // CHECK-NEXT: %obj = firrtl.object.create(%arg, %[[C]], %integer) {test.attr = "preserved"}
    %obj = firrtl.object {test.attr = "preserved"} @Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)
    %first = firrtl.object.subfield %obj[first] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    %duplicate = firrtl.object.subfield %obj[first] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %obj[result]
    %result = firrtl.object.subfield %obj[result] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    %constant = firrtl.object.subfield %obj[constant] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    %last = firrtl.object.subfield %obj[last] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %out, %[[RESULT]] : !firrtl.string
    firrtl.propassign %out, %result : !firrtl.string
    firrtl.propassign %last, %integer : !firrtl.integer
    %c = firrtl.integer 42
    firrtl.propassign %constant, %c : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %first, %arg : !firrtl.string
    // OM: om.object @Mixed(%{{.*}}, %arg, %{{.*}}, %integer)
  }

  // Object-reference cycles do not introduce hardware combinational loops.
  // Keep one declaration to break the SSA cycle; the other can still be lowered.
  // CHECK-LABEL: firrtl.module @Cycle
  // OM-LABEL: om.class @Cycle_Class
  firrtl.module @Cycle(out %out: !firrtl.anyref) {
    // CHECK-NEXT: %a = firrtl.object @Node
    // CHECK-NEXT: %[[A_REF:.*]] = firrtl.object.anyref_cast %a
    // CHECK-NEXT: %b = firrtl.object.create(%[[A_REF]])
    %a = firrtl.object @Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)

    %b = firrtl.object @Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)
    // CHECK-NEXT: %[[A_IN:.*]] = firrtl.object.subfield %a[ref]
    %a_in = firrtl.object.subfield %a[ref] : !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    %b_in = firrtl.object.subfield %b[ref] : !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    %a_ref = firrtl.object.anyref_cast %a : !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    // CHECK-NEXT: %[[B_REF:.*]] = firrtl.object.anyref_cast %b
    %b_ref = firrtl.object.anyref_cast %b : !firrtl.class<@Node(in ref: !firrtl.anyref, out next: !firrtl.anyref)>
    // CHECK-NEXT: firrtl.propassign %[[A_IN]], %[[B_REF]] : !firrtl.anyref
    // CHECK-NEXT: firrtl.propassign %out, %[[A_REF]] : !firrtl.anyref
    // CHECK-NEXT: }
    firrtl.propassign %a_in, %b_ref : !firrtl.anyref
    firrtl.propassign %b_in, %a_ref : !firrtl.anyref
    firrtl.propassign %out, %a_ref : !firrtl.anyref
  }

  // Classes remain unchanged, including the position of their constants.
  // CHECK-LABEL: firrtl.class @NoClassWires
  // OM-LABEL: om.class @NoClassWires
  firrtl.class @NoClassWires(in %arg: !firrtl.string, in %integer: !firrtl.integer) {
    // CHECK-NEXT: %obj = firrtl.object @Mixed
    %obj = firrtl.object @Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)
    // CHECK-NEXT: %[[FIRST:.*]] = firrtl.object.subfield %obj[first]
    %first = firrtl.object.subfield %obj[first] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    // CHECK-NEXT: %[[CONSTANT:.*]] = firrtl.object.subfield %obj[constant]
    %constant = firrtl.object.subfield %obj[constant] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    // CHECK-NEXT: %[[LAST:.*]] = firrtl.object.subfield %obj[last]
    %last = firrtl.object.subfield %obj[last] : !firrtl.class<@Mixed(in first: !firrtl.string, out result: !firrtl.string, in constant: !firrtl.integer, in last: !firrtl.integer)>
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 42
    %c = firrtl.integer 42
    // CHECK-NEXT: firrtl.propassign %[[FIRST]], %arg : !firrtl.string
    // CHECK-NEXT: firrtl.propassign %[[CONSTANT]], %[[C]] : !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %[[LAST]], %integer : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %first, %arg : !firrtl.string
    firrtl.propassign %constant, %c : !firrtl.integer
    firrtl.propassign %last, %integer : !firrtl.integer
  }

  // Modules without objects remain unchanged.
  // CHECK-LABEL: firrtl.module @NoObjects
  firrtl.module @NoObjects(out %out: !firrtl.integer) {
    // CHECK-NEXT: %[[WIRE:.*]] = firrtl.wire {{.*}}: !firrtl.integer
    %wire = firrtl.wire : !firrtl.integer
    // CHECK-NEXT: %[[C:.*]] = firrtl.integer 0
    %c = firrtl.integer 0
    // CHECK-NEXT: firrtl.propassign %[[WIRE]], %[[C]] : !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %out, %[[WIRE]] : !firrtl.integer
    // CHECK-NEXT: }
    firrtl.propassign %wire, %c : !firrtl.integer
    firrtl.propassign %out, %wire : !firrtl.integer
  }
}
