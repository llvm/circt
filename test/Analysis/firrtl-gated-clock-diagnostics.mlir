// RUN: circt-opt %s -test-firrtl-gated-clock-conversion -split-input-file -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -test-firrtl-gated-clock-conversion=verify-clock-aliases -split-input-file -verify-diagnostics -o /dev/null

// A local clock loop has no base clock.
firrtl.circuit "LocalClockFeedback" {
  firrtl.module @LocalClockFeedback(in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %w = firrtl.wire : !firrtl.clock
    // expected-warning @below {{this clock is not reachable from any free-running base clock}}
    %g = firrtl.int.clock_gate %w, %en
    firrtl.matchingconnect %w, %g : !firrtl.clock
    %r = firrtl.reg %g : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// A clock loop through two instances has no base clock.
firrtl.circuit "Top" {
  firrtl.module @ClockGen(in %clk_in: !firrtl.clock, in %en: !firrtl.uint<1>, out %clk_out: !firrtl.clock) {
    %g = firrtl.int.clock_gate %clk_in, %en
    firrtl.matchingconnect %clk_out, %g : !firrtl.clock
  }
  firrtl.module @Top(in %clk: !firrtl.clock, in %e1: !firrtl.uint<1>, in %e2: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %a_i, %a_e, %a_o = firrtl.instance g1 @ClockGen(in clk_in: !firrtl.clock, in en: !firrtl.uint<1>, out clk_out: !firrtl.clock)
    // expected-warning @below {{this clock is not reachable from any free-running base clock}}
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

// A mux of gated clocks has no single enable to sink.
firrtl.circuit "Top" {
  firrtl.module @Top(in %clk: !firrtl.clock, in %e1: !firrtl.uint<1>, in %e2: !firrtl.uint<1>, in %sel: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %g1 = firrtl.int.clock_gate %clk, %e1
    %g2 = firrtl.int.clock_gate %clk, %e2
    // expected-remark @below {{clock selection is not supported}}
    %m = firrtl.mux(%sel, %g1, %g2) : (!firrtl.uint<1>, !firrtl.clock, !firrtl.clock) -> !firrtl.clock
    %r = firrtl.reg %m : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// An ordinary clock mux is not reported.
firrtl.circuit "Top" {
  firrtl.module @Top(in %clkA: !firrtl.clock, in %clkB: !firrtl.clock, in %sel: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %m = firrtl.mux(%sel, %clkA, %clkB) : (!firrtl.uint<1>, !firrtl.clock, !firrtl.clock) -> !firrtl.clock
    %r = firrtl.reg %m : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d : !firrtl.uint<8>
  }
}

// -----

// An undriven clock is an error, raised before the IR is mutated.
firrtl.circuit "Top" {
  firrtl.module @Leaf(in %clk: !firrtl.clock, in %d: !firrtl.uint<8>) {
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
    // expected-warning @below {{expected exactly one connect driving this register}}
    %r = firrtl.reg %g : !firrtl.clock, !firrtl.uint<8>
  }
}

// -----

// A register with two connects is left alone, and adds no port pair to its
// module. An ungated register without a connect is not reported.
// CHECK-LABEL: firrtl.module @DeadPortsChild
// CHECK-NOT:     _gatedClock
// CHECK-LABEL: firrtl.module @DeadPorts(
// CHECK-NOT:     _gatedClock
// CHECK:       }
firrtl.circuit "DeadPorts" {
  firrtl.module @DeadPortsChild(in %clk: !firrtl.clock, in %d1: !firrtl.uint<8>, in %d2: !firrtl.uint<8>) {
    // expected-warning @below {{expected exactly one connect driving this register}}
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %r, %d1 : !firrtl.uint<8>
    firrtl.matchingconnect %r, %d2 : !firrtl.uint<8>
  }
  firrtl.module @DeadPorts(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>, in %d: !firrtl.uint<8>) {
    %g = firrtl.int.clock_gate %clk, %en
    %a_clk, %a_d1, %a_d2 = firrtl.instance a @DeadPortsChild(in clk: !firrtl.clock, in d1: !firrtl.uint<8>, in d2: !firrtl.uint<8>)
    firrtl.matchingconnect %a_clk, %g : !firrtl.clock
    firrtl.matchingconnect %a_d1, %d : !firrtl.uint<8>
    firrtl.matchingconnect %a_d2, %d : !firrtl.uint<8>
    %b_clk, %b_d1, %b_d2 = firrtl.instance b @DeadPortsChild(in clk: !firrtl.clock, in d1: !firrtl.uint<8>, in d2: !firrtl.uint<8>)
    firrtl.matchingconnect %b_clk, %clk : !firrtl.clock
    firrtl.matchingconnect %b_d1, %d : !firrtl.uint<8>
    firrtl.matchingconnect %b_d2, %d : !firrtl.uint<8>
    %r = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
  }
}
