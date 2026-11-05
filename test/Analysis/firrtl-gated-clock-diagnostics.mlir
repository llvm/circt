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

// -----

// Without a connect there is nowhere to sink the enable.
firrtl.circuit "Top" {
  firrtl.module @Top(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>) {
    %g = firrtl.int.clock_gate %clk, %en
    // expected-error @below {{expected exactly one connect driving this register (run after firrtl-expand-whens); found 0}}
    %r = firrtl.reg %g : !firrtl.clock, !firrtl.uint<8>
  }
}

// -----

// A register with two connects is rejected, even if its clock is not gated.
firrtl.circuit "TwoConnects" {
  firrtl.module @TwoConnects(in %clk: !firrtl.clock, in %d1: !firrtl.uint<8>, in %d2: !firrtl.uint<8>) {
    // expected-error @below {{expected exactly one connect driving this register (run after firrtl-expand-whens); found 2}}
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d1 : !firrtl.uint<8>
    firrtl.matchingconnect %r, %d2 : !firrtl.uint<8>
  }
}

// -----

// The enable cannot be sunk into an aggregate register.
firrtl.circuit "AggregateReg" {
  firrtl.module @AggregateReg(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                              in %d: !firrtl.bundle<a: uint<8>>) {
    %g = firrtl.int.clock_gate %clk, %en
    // expected-error @below {{expected a ground-typed register (run after firrtl-lower-types); found '!firrtl.bundle<a: uint<8>>'}}
    %r = firrtl.reg %g : !firrtl.clock, !firrtl.bundle<a: uint<8>>
    firrtl.matchingconnect %r, %d : !firrtl.bundle<a: uint<8>>
  }
}
