// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-promote-objects)))' | FileCheck %s

firrtl.circuit "Objects" {
  firrtl.class @Pair(in %first: !firrtl.string, out %result: !firrtl.string, in %second: !firrtl.integer) {
    firrtl.propassign %result, %first : !firrtl.string
  }
  firrtl.extclass @Empty()

  // CHECK-LABEL: firrtl.module @Objects
  firrtl.module @Objects(in %arg: !firrtl.string, out %out: !firrtl.string, out %ref: !firrtl.anyref) {
    // Check input promotion and object access handling.
    // CHECK-NEXT: %[[I:.*]] = firrtl.integer 42
    // CHECK-NEXT: %pair = firrtl.object.create(%arg, %[[I]]) : (!firrtl.string, !firrtl.integer) -> !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %pair = firrtl.object @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %first = firrtl.object.subfield %pair[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %duplicate = firrtl.object.subfield %pair[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %pair[result]
    %result = firrtl.object.subfield %pair[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %second = firrtl.object.subfield %pair[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %int = firrtl.integer 42
    firrtl.propassign %second, %int : !firrtl.integer
    firrtl.propassign %first, %arg : !firrtl.string
    // Preserve output accesses and other object uses.
    // CHECK-NEXT: firrtl.propassign %out, %[[RESULT]] : !firrtl.string
    firrtl.propassign %out, %result : !firrtl.string
    // CHECK-NEXT: %[[REF:.*]] = firrtl.object.anyref_cast %pair
    %cast = firrtl.object.anyref_cast %pair : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: firrtl.propassign %ref, %[[REF]] : !firrtl.anyref
    firrtl.propassign %ref, %cast : !firrtl.anyref
    // CHECK-NEXT: %empty = firrtl.object.create() : () -> !firrtl.class<@Empty()>
    // CHECK-NEXT: }
    %empty = firrtl.object @Empty()
  }
}
