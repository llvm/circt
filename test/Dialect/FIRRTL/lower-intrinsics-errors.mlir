// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' -verify-diagnostics --split-input-file %s

firrtl.circuit "UnknownIntrinsic" {
  firrtl.module private @UnknownIntrinsic(in %data: !firrtl.uint<32>) {
    %0 = firrtl.wire : !firrtl.uint<32>
    // expected-error @below {{unknown intrinsic}}
    // expected-error @below {{failed to legalize}}
    firrtl.int.generic "unknown_intrinsic" %0 : (!firrtl.uint<32>) -> ()
    firrtl.strictconnect %0, %data : !firrtl.uint<32>
  }
}

// -----

firrtl.circuit "InvalidCGOperand" {
    firrtl.module @InvalidCGOperand(in %clk: !firrtl.clock, in %en: !firrtl.uint<2>) {
      // expected-error @below {{circt.clock_gate input 1 not size 1}}
      // expected-error @below {{failed to legalize}}
      %0 = firrtl.int.generic "circt.clock_gate" %clk, %en : (!firrtl.clock, !firrtl.uint<2>) -> !firrtl.clock
    }
}

