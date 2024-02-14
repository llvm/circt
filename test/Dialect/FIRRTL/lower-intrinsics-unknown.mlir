// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics{fixup-eicg-wrapper}))' -verify-diagnostics %s

firrtl.circuit "UnknownIntrinsic" {
  // expected-error @below {{intrinsic not recognized}}
  firrtl.intmodule private @UnknownIntrinsicMod(in data: !firrtl.uint<32>) attributes {intrinsic = "unknown_intrinsic"}

  firrtl.module private @UnknownIntrinsic(in %data : !firrtl.uint<32>) {
    %mod_data = firrtl.instance mod @UnknownIntrinsicMod(in data: !firrtl.uint<32>)
    firrtl.strictconnect %mod_data, %data : !firrtl.uint<32>
  }
}