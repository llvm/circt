// RUN: circt-opt --ibis-containerize %s | FileCheck %s

// CHECK-LABEL:   ibis.container.outer @A_B
// CHECK-LABEL:   ibis.container.outer @MyClass
// CHECK-LABEL:   ibis.container.outer @A_B_0
// CHECK-LABEL:   ibis.container.outer @A_C
// CHECK-LABEL:   ibis.container.outer @A {
// CHECK:           %[[VAL_0:.*]] = ibis.this @A
// CHECK:           ibis.port.input @A_in : i1
// CHECK:           %[[VAL_1:.*]] = ibis.container.instance @myClass, @MyClass
// CHECK:           %[[VAL_2:.*]] = ibis.container.instance @A_B_0, @A_B_0
// CHECK:           %[[VAL_3:.*]] = ibis.container.instance @A_C, @A_C

// This container will alias with the @B inside @A, and thus checks the
// name uniquing logic.
ibis.container.outer @A_B {
  %this = ibis.this @A_B
}

ibis.class @MyClass {
  %this = ibis.this @MyClass
}

ibis.class @A {
  %this = ibis.this @A
  ibis.port.input @A_in : i1
  %myClass = ibis.instance @myClass, @MyClass
  ibis.container.inner @B {
    %B_this = ibis.this @B
  }
  ibis.container.inner @C {
    %C_this = ibis.this @C
  }
}
