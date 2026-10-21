// RUN: circt-opt %s -firrtl-lower-objects | FileCheck %s
// RUN: circt-opt %s -firrtl-lower-objects -firrtl-lower-objects | FileCheck %s
// RUN: circt-opt %s -firrtl-lower-objects -firrtl-lower-classes | FileCheck %s --check-prefix=OM

firrtl.circuit "Objects" {
  firrtl.class @Pair(in %first: !firrtl.string, out %result: !firrtl.string, in %second: !firrtl.integer) {
    firrtl.propassign %result, %first : !firrtl.string
  }
  firrtl.extclass @Empty()

  // CHECK-LABEL: firrtl.module @Objects
  // OM-LABEL: om.class @Objects_Class
  firrtl.module @Objects(out %out: !firrtl.string, out %ref: !firrtl.anyref) {
    // The values are defined after the object, and assignments are in reverse
    // input order. The lowering must establish SSA dominance and port order.
    %pair = firrtl.object {test.attr = "preserved"} @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %first = firrtl.object.subfield %pair[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %result = firrtl.object.subfield %pair[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %second = firrtl.object.subfield %pair[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK: %[[S:.*]] = firrtl.string "hello"
    // CHECK-NEXT: %[[I:.*]] = firrtl.integer 42
    // CHECK-NEXT: %empty = firrtl.object.create() : () -> !firrtl.class<@Empty()>
    // CHECK-NEXT: %pair = firrtl.object.create(%[[S]], %[[I]]) {test.attr = "preserved"} : (!firrtl.string, !firrtl.integer) -> !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %pair[result]
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
    %empty = firrtl.object @Empty()
  }

  // Test dependencies between objects and objects nested in a class.
  // CHECK-LABEL: firrtl.class @Chain
  // OM-LABEL: om.class @Chain
  firrtl.class @Chain(in %arg: !firrtl.string, out %out: !firrtl.string) {
    %a = firrtl.object @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %b = firrtl.object @Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)
    %a_in = firrtl.object.subfield %a[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %a_int = firrtl.object.subfield %a[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %a_out = firrtl.object.subfield %a[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %b_in = firrtl.object.subfield %b[first] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %b_int = firrtl.object.subfield %b[second] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    %b_out = firrtl.object.subfield %b[result] : !firrtl.class<@Pair(in first: !firrtl.string, out result: !firrtl.string, in second: !firrtl.integer)>
    // CHECK: %[[I:.*]] = firrtl.integer 0
    // CHECK-NEXT: %b = firrtl.object.create(%arg, %[[I]])
    // CHECK-NEXT: %[[BOUT:.*]] = firrtl.object.subfield %b[result]
    // CHECK-NEXT: %a = firrtl.object.create(%[[BOUT]], %[[I]])
    // OM: %[[B:.*]] = om.object @Pair
    // OM: %[[BOUT:.*]] = om.object.field %[[B]]["result"]
    // OM: om.object @Pair(%{{.*}}, %[[BOUT]], %{{.*}})
    %int = firrtl.integer 0
    firrtl.propassign %a_in, %b_out : !firrtl.string
    firrtl.propassign %b_in, %arg : !firrtl.string
    firrtl.propassign %a_int, %int : !firrtl.integer
    firrtl.propassign %b_int, %int : !firrtl.integer
    firrtl.propassign %out, %a_out : !firrtl.string
  }
}
