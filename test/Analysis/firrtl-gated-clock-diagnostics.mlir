// RUN: circt-opt %s -test-firrtl-gated-clock-conversion -split-input-file -verify-diagnostics

// A local clock loop has no base clock.
firrtl.circuit "LocalClockFeedback" {
  firrtl.module @LocalClockFeedback(in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %w = firrtl.wire : !firrtl.clock
    // expected-error @below {{this clock is driven by a clock loop, and is not reachable from any base clock}}
    %g = firrtl.int.clock_gate %w, %en
    firrtl.matchingconnect %w, %g : !firrtl.clock
    %r = firrtl.reg %g : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// A clock loop through two instances has no base clock.
firrtl.circuit "Top" {
  firrtl.module private @ClockGen(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
    %g = firrtl.int.clock_gate %clk_in, %en
    firrtl.matchingconnect %clk_out, %g : !firrtl.clock
  }
  firrtl.module @Top(in %clk: !firrtl.clock, in %e1: !firrtl.uint<1>, in %e2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %a_i, %a_e, %a_o = firrtl.instance g1 @ClockGen(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    // expected-error @below {{this clock is driven by a clock loop, and is not reachable from any base clock}}
    %b_i, %b_e, %b_o = firrtl.instance g2 @ClockGen(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    firrtl.matchingconnect %a_i, %b_o : !firrtl.clock
    firrtl.matchingconnect %a_e, %e1 : !firrtl.uint<1>
    firrtl.matchingconnect %b_i, %a_o : !firrtl.clock
    firrtl.matchingconnect %b_e, %e2 : !firrtl.uint<1>
    %r = firrtl.reg %b_o : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// A gated caller gives the child's clock port a port pair, which a caller
// whose clock is in a clock loop cannot drive.
firrtl.circuit "Top" {
  firrtl.module private @Child(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
  firrtl.module @Top(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %w = firrtl.wire : !firrtl.clock
    %loop = firrtl.int.clock_gate %w, %en
    firrtl.matchingconnect %w, %loop : !firrtl.clock
    // expected-error @below {{clock port "clk" of this instance is driven by a clock loop, and is not reachable from any base clock}}
    %a_clk, %a_d = firrtl.instance a @Child(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %a_clk, %loop : !firrtl.clock
    firrtl.matchingconnect %a_d, %d : !firrtl.uint<8>
    %g = firrtl.int.clock_gate %clk, %en
    %b_clk, %b_d = firrtl.instance b @Child(in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %b_clk, %g : !firrtl.clock
    firrtl.matchingconnect %b_d, %d : !firrtl.uint<8>
  }
}

// -----

// An output-only port pair cannot be added to an instance_choice.
firrtl.circuit "Top" {
  firrtl.option @Opt { firrtl.option_case @X }
  firrtl.module private @Gen(in %u: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %o: !firrtl.clock) {
    %c = firrtl.asClock %u : (!firrtl.uint<1>) -> !firrtl.clock
    %g = firrtl.int.clock_gate %c, %en
    firrtl.matchingconnect %o, %g : !firrtl.clock
  }
  firrtl.module @Top(in %u: !firrtl.uint<1>, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %i_u, %i_en, %i_o = firrtl.instance i @Gen(in u: !firrtl.uint<1>, in en: !firrtl.uint<1>, out o: !firrtl.clock)
    firrtl.matchingconnect %i_u, %u : !firrtl.uint<1>
    firrtl.matchingconnect %i_en, %en : !firrtl.uint<1>
    %r = firrtl.reg %i_o : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    // expected-error @below {{can only handle InstanceOp}}
    %c_u, %c_en, %c_o = firrtl.instance_choice c @Gen alternatives @Opt { @X -> @Gen } (in u: !firrtl.uint<1>, in en: !firrtl.uint<1>, out o: !firrtl.clock)
    firrtl.matchingconnect %c_u, %u : !firrtl.uint<1>
    firrtl.matchingconnect %c_en, %en : !firrtl.uint<1>
  }
}

// -----

// An input port pair cannot be added to an instance_choice. It is a use of
// `@Child` once per target, and is reported once.
firrtl.circuit "Top" {
  firrtl.option @Opt { firrtl.option_case @X }
  firrtl.module private @Child(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
  firrtl.module @Top(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %en
    // expected-error @below {{can only handle InstanceOp}}
    %c_clk, %c_d = firrtl.instance_choice c @Child alternatives @Opt { @X -> @Child } (in clk: !firrtl.clock, in d: !firrtl.uint<8>)
    firrtl.matchingconnect %c_clk, %g : !firrtl.clock
    firrtl.matchingconnect %c_d, %d : !firrtl.uint<8>
  }
}

// -----

// An instance_choice of a module with an output port pair is reported once,
// even though `analyze()` also traces its input clock.
firrtl.circuit "Top" {
  firrtl.option @Opt { firrtl.option_case @X }
  firrtl.module private @Gen(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, out %o: !firrtl.clock) {
    %g = firrtl.int.clock_gate %clk, %en
    firrtl.matchingconnect %o, %g : !firrtl.clock
  }
  firrtl.module @Top(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %i_clk, %i_en, %i_o = firrtl.instance i @Gen(in clk: !firrtl.clock, in en: !firrtl.uint<1>, out o: !firrtl.clock)
    firrtl.matchingconnect %i_clk, %clk : !firrtl.clock
    firrtl.matchingconnect %i_en, %en : !firrtl.uint<1>
    %r = firrtl.reg %i_o : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
    // expected-error @below {{can only handle InstanceOp}}
    %c_clk, %c_en, %c_o = firrtl.instance_choice c @Gen alternatives @Opt { @X -> @Gen } (in clk: !firrtl.clock, in en: !firrtl.uint<1>, out o: !firrtl.clock)
    firrtl.matchingconnect %c_clk, %clk : !firrtl.clock
    firrtl.matchingconnect %c_en, %en : !firrtl.uint<1>
  }
}

// -----

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

// Whether the enable must gate the reset depends on whether it is sync.
firrtl.circuit "AbstractReset" {
  firrtl.module @AbstractReset(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>,
                               in %rst: !firrtl.reset, in %d: !firrtl.uint<8>,
                               in %init: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %en
    // expected-error @below {{register has an uninferred reset type (run after firrtl-infer-resets)}}
    %r = firrtl.regreset %g, %rst, %init : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
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
