// RUN: circt-opt --pass-pipeline='builtin.module(ibis.design(ibis-containerize))' %s | FileCheck %s

ibis.design @foo {

// CHECK-LABEL:   ibis.container sym @A_B
// CHECK-LABEL:   ibis.container "MyClassName" sym @MyClass
// CHECK-LABEL:   ibis.container sym @A_B_0
// CHECK-LABEL:   ibis.container sym @A_C
// CHECK-LABEL:   ibis.container sym @A {
// CHECK:           %[[VAL_0:.*]] = ibis.this <@foo::@A>
// CHECK:           ibis.port.input "A_in" sym @A_in : i1
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @myClass, <@foo::@MyClass>
// CHECK:           %[[VAL_2:.*]] = ibis.container.instance @A_B_0, <@foo::@A_B_0>
// CHECK:           %[[VAL_3:.*]] = ibis.container.instance @A_C, <@foo::@A_C>

// This container will alias with the @B inside @A, and thus checks the
// name uniquing logic.
ibis.container sym @A_B {
  %this = ibis.this <@foo::@A_B>
}

ibis.class "MyClassName" sym @MyClass {
  %this = ibis.this <@foo::@MyClass>
}

ibis.class sym @A {
  %this = ibis.this <@foo::@A>
  ibis.port.input "A_in" sym @A_in : i1
  %myClass = ibis.instance @myClass, <@foo::@MyClass>
  ibis.container sym @B {
    %B_this = ibis.this <@foo::@B>
  }
  ibis.container sym @C {
    %C_this = ibis.this <@foo::@C>
  }
}

}
