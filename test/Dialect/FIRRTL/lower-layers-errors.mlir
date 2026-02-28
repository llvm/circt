// RUN: circt-opt -firrtl-lower-layers -split-input-file -verify-diagnostics %s

firrtl.circuit "NonPassiveSubaccess" {
  firrtl.layer @A bind {}
  firrtl.module @NonPassiveSubaccess(
    in %a: !firrtl.vector<bundle<a: uint<1>, b flip: uint<1>>, 2>,
    in %b: !firrtl.uint<1>
  ) {
    // expected-note @below {{the layerblock is defined here}}
    firrtl.layerblock @A {
      %n = firrtl.node %b : !firrtl.uint<1>
      // expected-error @below {{'firrtl.subaccess' op has a non-passive operand and captures a value defined outside its enclosing bind-convention layerblock}}
      %0 = firrtl.subaccess %a[%n] : !firrtl.vector<bundle<a: uint<1>, b flip: uint<1>>, 2>, !firrtl.uint<1>
    }
  }
}
