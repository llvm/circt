// RUN: circt-opt --ibis-containerize %s | FileCheck %s

// CHECK-LABEL:   ibis.container @A_B
// CHECK-LABEL:   ibis.container @A_B_0
// CHECK-LABEL:   ibis.container @A_C
// CHECK-LABEL:   ibis.container @A {
// CHECK:           %[[VAL_0:.*]] = ibis.this @A
// CHECK:           ibis.port.input @A_in : i1
// CHECK:         }

// This container will alias with the @B inside @A, and thus checks the
// name uniquing logic.
ibis.container @A_B {
  %this = ibis.this @A_B
}

ibis.class @A {
  %this = ibis.this @A
  ibis.port.input @A_in : i1
  ibis.container @B {
    %B_this = ibis.this @B
  }
  ibis.container @C {
    %C_this = ibis.this @C
  }
}
