// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects,firrtl-lower-objects)))' | FileCheck %s
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-lower-objects),firrtl-lower-classes))' | FileCheck %s --check-prefix=OM

firrtl.circuit "Objects" {
  firrtl.class @Pair(in %first: !firrtl.string, out %result: !firrtl.string, in %second: !firrtl.integer) {
    firrtl.propassign %result, %first : !firrtl.string
  }
  firrtl.extclass @Empty()

  // CHECK-LABEL: firrtl.module @Objects
  // OM-LABEL: om.class @Objects_Class
  firrtl.module @Objects(out %out: !firrtl.string, out %ref: !firrtl.anyref) {
    // The values are defined after the object, and assignments are in reverse
    // input order. Hoist constants and create the object at its declaration.
    // CHECK-NEXT: %[[S:.*]] = firrtl.string "hello"
    // CHECK-NEXT: %[[I:.*]] = firrtl.integer 42
    // CHECK-NEXT: %pair = firrtl.object.create(%[[S]], %[[I]]) {test.attr = "preserved"} : (!firrtl.string, !firrtl.integer) -> !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %pair[result]
    %pair = firrtl.object {test.attr = "preserved"} @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %first = firrtl.object.subfield %pair[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %result = firrtl.object.subfield %pair[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %second = firrtl.object.subfield %pair[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // OM: %[[S:.*]] = om.constant "hello"
    // OM: %[[I:.*]] = om.constant #om.integer<42
    // OM: om.object @Pair(%{{.*}}, %[[S]], %[[I]])
    %str = firrtl.string "hello"
    %int = firrtl.integer 42
    firrtl.propassign %second, %int : !firrtl.integer
    firrtl.propassign %first, %str : !firrtl.string
    // CHECK: firrtl.propassign %out, %[[RESULT]] : !firrtl.string
    firrtl.propassign %out, %result : !firrtl.string
    // CHECK: %[[REF:.*]] = firrtl.object.anyref_cast %pair
    %cast = firrtl.object.anyref_cast %pair : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    firrtl.propassign %ref, %cast : !firrtl.anyref
    // No-input objects can be replaced directly as well.
    // CHECK: %empty = firrtl.object.create() : () -> !firrtl.class<@Empty()>
    // CHECK-NEXT: }
    %empty = firrtl.object @Empty()
  }

  // Objects inside a class remain in declaration form.
  // CHECK-LABEL: firrtl.class @Chain
  // OM-LABEL: om.class @Chain
  firrtl.class @Chain(in %arg: !firrtl.string, out %out: !firrtl.string) {
    // CHECK-NEXT: %a = firrtl.object @Pair
    // CHECK-NEXT: %b = firrtl.object @Pair
    %a = firrtl.object @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %b = firrtl.object @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    // CHECK-NEXT: %[[AIN:.*]] = firrtl.object.subfield %a[first]
    %a_in = firrtl.object.subfield %a[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[AINT:.*]] = firrtl.object.subfield %a[second]
    %a_int = firrtl.object.subfield %a[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[AOUT:.*]] = firrtl.object.subfield %a[result]
    %a_out = firrtl.object.subfield %a[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[BIN:.*]] = firrtl.object.subfield %b[first]
    %b_in = firrtl.object.subfield %b[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[BINT:.*]] = firrtl.object.subfield %b[second]
    %b_int = firrtl.object.subfield %b[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[BOUT:.*]] = firrtl.object.subfield %b[result]
    %b_out = firrtl.object.subfield %b[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[I:.*]] = firrtl.integer 0
    // CHECK-NEXT: firrtl.propassign %[[AIN]], %[[BOUT]] : !firrtl.string
    // CHECK-NEXT: firrtl.propassign %[[BIN]], %arg : !firrtl.string
    // CHECK-NEXT: firrtl.propassign %[[AINT]], %[[I]] : !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %[[BINT]], %[[I]] : !firrtl.integer
    // CHECK-NEXT: firrtl.propassign %out, %[[AOUT]] : !firrtl.string
    // OM-DAG: %[[B:.*]] = om.object @Pair(%{{.*}}, %arg, %{{.*}})
    // OM-DAG: %[[BOUT:.*]] = om.object.field %[[B]]["result"]
    // OM-DAG: om.object @Pair(%{{.*}}, %[[BOUT]], %{{.*}})
    %int = firrtl.integer 0
    firrtl.propassign %a_in, %b_out : !firrtl.string
    firrtl.propassign %b_in, %arg : !firrtl.string
    firrtl.propassign %a_int, %int : !firrtl.integer
    firrtl.propassign %b_int, %int : !firrtl.integer
    firrtl.propassign %out, %a_out : !firrtl.string
  }
}
