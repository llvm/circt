// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL:  ibis.container @A {
// CHECK-NEXT:    ibis.port.input @A_in : i1
// CHECK-NEXT:    ibis.port.output @A_out : i1
// CHECK-NEXT:    ibis.container @B {
// CHECK-NEXT:      ibis.port.input @B_in : i1
// CHECK-NEXT:      ibis.port.output @B_out : i1
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-LABEL:  ibis.container @C {
// CHECK-NEXT:    ibis.port.input @C_in : i1
// CHECK-NEXT:    ibis.port.output @C_out : i1
// CHECK-NEXT:    ibis.container.instance @My_A, @A
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    ibis.port.write @C_in(%true) : i1
// CHECK-NEXT:    %0 = ibis.port.read @C_out : i1
// CHECK-NEXT:    ibis.port.write @My_A::@A_in(%0) : i1
// CHECK-NEXT:    %1 = ibis.port.read @My_A::@A_out : i1
// CHECK-NEXT:    ibis.container @D {
// CHECK-NEXT:      ibis.container @E {
// CHECK-NEXT:        %true_0 = hw.constant true
// CHECK-NEXT:        ibis.port.write @C_in(%true_0) : i1
// CHECK-NEXT:        %2 = ibis.port.read @C_out : i1
// CHECK-NEXT:        ibis.port.write @My_A::@A_in(%2) : i1
// CHECK-NEXT:        %3 = ibis.port.read @My_A::@A_out : i1
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }

ibis.class @A {
  ibis.port.input @A_in : i1
  ibis.port.output @A_out : i1
}

ibis.class @C {
  ibis.port.input @C_in : i1
  ibis.port.output @C_out : i1

  ibis.instance @My_A, @A

  // Test local read/writes
  %true = hw.constant true
  ibis.port.write @C_in(%true) : i1
  %c_out = ibis.port.read @C_out : i1

  // Test cross-container read/writes
  ibis.port.write @My_A::@A_in(%c_out) : i1
  %a_out = ibis.port.read @My_A::@A_out : i1

  ibis.container @D {
    // Test parent read/writes
    %true_1 = hw.constant true
    ibis.port.write @C_in(%true_1) : i1
    %c_out_1 = ibis.port.read @C_out : i1

    // Test parent instance read/writes
    ibis.port.write @My_A::@A_in(%c_out_1) : i1
    %a_out_1 = ibis.port.read @My_A::@A_out : i1
  }
}
