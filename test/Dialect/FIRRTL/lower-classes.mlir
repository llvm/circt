// RUN: circt-opt -firrtl-lower-classes %s | FileCheck %s

firrtl.circuit "Component" {
  // CHECK-LABEL: om.class @Class_0
  // CHECK-SAME: %[[REF1:.+]]: !om.class.type<@Class_1>
  firrtl.class private @Class_0(in %someReference_in: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>, out %someReference: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>) {
    // CHECK: om.class.field @someReference, %[[REF1]]
    firrtl.propassign %someReference, %someReference_in : !firrtl.class<@Class_1(out someInt: !firrtl.integer)>
  }

  // CHECK-LABEL: om.class @Class_1
  firrtl.class private @Class_1(out %someInt: !firrtl.integer) {
    // CHECK: %[[C1:.+]] = om.constant 1 : si4
    %0 = firrtl.integer 1
    // CHECK: om.class.field @someInt, %[[C1]]
    firrtl.propassign %someInt, %0 : !firrtl.integer
  }

  // CHECK-LABEL: om.class @Class_2
  firrtl.class private @Class_2(out %someString: !firrtl.string) {
    // CHECK: %[[C2:.+]] = om.constant "fubar" : !om.string
    %0 = firrtl.string "fubar"
    // CHECK: om.class.field @someString, %[[C2]]
    firrtl.propassign %someString, %0 : !firrtl.string
  }

  // CHECK-LABEL: om.class @ClassEntrypoint
  firrtl.class private @ClassEntrypoint(out %obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>) {
    // CHECK: %[[OBJ1:.+]] = om.object @Class_1() : () -> !om.class.type<@Class_1>
    %0 = firrtl.object @Class_1(out someInt: !firrtl.integer)
    // CHECK: om.class.field @obj_0_out, %[[OBJ1]]
    firrtl.propassign %obj_0_out, %0 : !firrtl.class<@Class_1(out someInt: !firrtl.integer)>

    // TODO: instantiate Class_0 and pass the reference to Class_1 in once object subfield flows are sorted.
  }

  firrtl.module @Component(in %input: !firrtl.uint<1>, out %output: !firrtl.uint<1>, out %omir_out: !firrtl.class<@ClassEntrypoint(out obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.object @ClassEntrypoint(out obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)
    firrtl.propassign %omir_out, %0 : !firrtl.class<@ClassEntrypoint(out obj_0_out: !firrtl.class<@Class_1(out someInt: !firrtl.integer)>)>
    firrtl.strictconnect %output, %input : !firrtl.uint<1>
  }
}
