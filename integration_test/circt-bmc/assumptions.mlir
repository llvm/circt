// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

// Register r starts at 1 and samples input in; the assumption forces in == 1,
// so r can never become 0. Assumptions must keep holding for past timesteps
// when the solver checks later ones; if they are dropped along with the
// per-step assertions, this reports a spurious violation.
//  RUN: circt-bmc %s -b 10 --module AssumeDrop --shared-libs=%libz3 | FileCheck %s --check-prefix=ASSUMEDROP
//  ASSUMEDROP: Bound reached with no violations!
hw.module @AssumeDrop(in %clk: !seq.clock, in %in: i1) {
  %init = seq.initial () {
    %c1 = hw.constant true
    seq.yield %c1 : i1
  } : () -> !seq.immutable<i1>
  %r = seq.compreg %in, %clk initial %init : i1
  verif.assume %in : i1
  verif.assert %r : i1
}

// Same design without the assumption: r becomes 0 one cycle after in == 0.
//  RUN: circt-bmc %s -b 10 --module Violation --shared-libs=%libz3 | FileCheck %s --check-prefix=VIOLATION
//  VIOLATION: Assertion can be violated!
hw.module @Violation(in %clk: !seq.clock, in %in: i1) {
  %init = seq.initial () {
    %c1 = hw.constant true
    seq.yield %c1 : i1
  } : () -> !seq.immutable<i1>
  %r = seq.compreg %in, %clk initial %init : i1
  verif.assert %r : i1
}

// Same design with the input hardwired to 1 instead of assumed.
//  RUN: circt-bmc %s -b 10 --module Control --shared-libs=%libz3 | FileCheck %s --check-prefix=CONTROL
//  CONTROL: Bound reached with no violations!
hw.module @Control(in %clk: !seq.clock) {
  %c1 = hw.constant true
  %init = seq.initial () {
    %c1i = hw.constant true
    seq.yield %c1i : i1
  } : () -> !seq.immutable<i1>
  %r = seq.compreg %c1, %clk initial %init : i1
  verif.assert %r : i1
}
