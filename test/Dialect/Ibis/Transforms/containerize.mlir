// RUN: circt-opt --pass-pipeline='builtin.module(ibis.design(ibis-containerize))' %s | FileCheck %s

ibis.design @foo {

// CHECK-LABEL:   ibis.container @A_B
// CHECK-LABEL:   ibis.container @MyClass
// CHECK-LABEL:   ibis.container @A_B_0
// CHECK-LABEL:   ibis.container @A_C
// CHECK-LABEL:   ibis.container @A {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@A>
// CHECK:           ibis.port.input "A_in" sym @A_in : i1
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @myClass, <@MyClass>
// CHECK:           %[[VAL_2:.*]] = ibis.container.instance @A_B_0, <@A_B_0>
// CHECK:           %[[VAL_3:.*]] = ibis.container.instance @A_C, <@A_C>

// This container will alias with the @B inside @A, and thus checks the
// name uniquing logic.
ibis.container @A_B {
  %this = ibis.this <@A_B>
}

ibis.class @MyClass {
  %this = ibis.this <@MyClass>
}

ibis.class @A {
  %this = ibis.this <@A>
  ibis.port.input "A_in" sym @A_in : i1
  %myClass = ibis.instance @myClass, <@MyClass>
  ibis.container @B {
    %B_this = ibis.this <@A::@B>
  }
  ibis.container @C {
    %C_this = ibis.this <@A::@C>
  }
}

}
