// RUN: circt-opt %s --arc-split-loops --verify-diagnostics --split-input-file

hw.module @UnbreakableLoop(in %clock : !seq.clock, in %a : i4, out x : i4) {
  // expected-error @below {{loop splitting did not eliminate all loops; loop detected}}
  // expected-note @below {{through operand 1 here:}}
  %0, %1 = arc.call @UnbreakableLoopArc(%a, %0) : (i4, i4) -> (i4, i4)
  hw.output %1 : i4
}

arc.define @UnbreakableLoopArc(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %true = hw.constant true
  %0:2 = scf.if %true -> (i4, i4) {
    scf.yield %arg0, %arg1 : i4, i4
  } else {
    scf.yield %arg1, %arg0 : i4, i4
  }
  arc.output %0#0, %0#1 : i4, i4
}

// -----
// An unclocked `sim.func.dpi.call` is not arc-breaking and must not be allowed
// to hide a feedback loop through arcs.

sim.func.dpi @UnclockedDpi(in %arg0: i4, return ret: i4)

hw.module @UnclockedDpiNoBreakLoop(in %a: i4, out x: i4, out y: i4) {
  // expected-error @below {{loop splitting did not eliminate all loops; loop detected}}
  // expected-note @below {{through operand 0 here:}}
  %1 = sim.func.dpi.call @UnclockedDpi(%0#0) : (i4) -> i4
  // expected-note @below {{through operand 1 here:}}
  %0:2 = arc.call @UnclockedDpiArc(%a, %1) : (i4, i4) -> (i4, i4)
  hw.output %0#0, %0#1 : i4, i4
}

arc.define @UnclockedDpiArc(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %0 = comb.add %arg0, %arg1 : i4
  %1 = comb.mul %arg1, %arg0 : i4
  arc.output %0, %1 : i4, i4
}
