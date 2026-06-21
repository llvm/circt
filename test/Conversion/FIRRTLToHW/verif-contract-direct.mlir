// RUN: circt-opt --lower-firrtl-to-hw --verify-diagnostics %s | FileCheck %s

// A `verif.contract` emitted directly into a FIRRTL module body (instead of via
// `firrtl.contract`) must survive FIRRTL-to-HW lowering: its FIRRTL-typed
// operands/results are lowered, and its body is re-lowered in place. The
// frontend inserts the property type conversion (firrtl.uint<1> -> i1) so that
// verif.require/verif.ensure only ever see core types.

firrtl.circuit "VerifContractDirect" {
  // CHECK-LABEL: hw.module @VerifContractDirect
  // CHECK-NEXT:    [[R:%.+]] = verif.contract %a : i42 {
  // CHECK-NEXT:      verif.require %p : i1
  // CHECK-NEXT:      verif.ensure %p : i1
  // CHECK-NEXT:    }
  // CHECK-NEXT:    hw.output [[R]] : i42
  
  firrtl.module @VerifContractDirect(in %a: !firrtl.uint<42>, in %p: !firrtl.uint<1>, out %b: !firrtl.uint<42>) {
    %0 = verif.contract %a : !firrtl.uint<42> {
      %p_i1 = builtin.unrealized_conversion_cast %p : !firrtl.uint<1> to i1
      verif.require %p_i1 : i1
      verif.ensure %p_i1 : i1
    }
    firrtl.matchingconnect %b, %0 : !firrtl.uint<42>
  }
}
