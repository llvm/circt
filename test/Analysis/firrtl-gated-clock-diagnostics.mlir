// RUN: circt-opt %s -test-firrtl-gated-clock-conversion -split-input-file -verify-diagnostics


// An undriven clock is an error, raised before the IR is mutated.
firrtl.circuit "Top" {
  firrtl.module private @Leaf(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
  firrtl.module @Top(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %en
    %a_clk, %a_d = firrtl.instance a @Leaf(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %a_clk, %g : !firrtl.clock
    firrtl.matchingconnect %a_d, %d : !firrtl.uint<8>
    // expected-error @below {{this clock is not driven}}
    %b_clk, %b_d = firrtl.instance b @Leaf(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %b_d, %d : !firrtl.uint<8>
  }
}
