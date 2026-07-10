// RUN: circt-opt %s --split-input-file --verify-diagnostics

firrtl.circuit "Test" {
  firrtl.module @Test(in %a : !firrtl.uint<1>) {
    // expected-error @+1 {{attribute 'delay' failed to satisfy constraint}}
    %0 = firrtl.int.ltl.delay %a, -1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Test" {
  firrtl.module @Test(in %a : !firrtl.uint<1>) {
    // expected-error @+1 {{attribute 'length' failed to satisfy constraint}}
    %0 = firrtl.int.ltl.delay %a, 1, -1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  }
}
