// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

hw.module @Foo(in %a: i42, in %clk: !seq.clock) {
  // CHECK: seq.compreg sym @"foo/bar"
  %0 = seq.compreg sym @"foo/bar" %a, %clk : i42
  hw.output
}

hw.module @Bar(in %d: i32, in %clk: !seq.clock, in %en: i1) {
  // CHECK: seq.compreg.ce sym @"weird name"
  %1 = seq.compreg.ce sym @"weird name" %d, %clk, %en : i32
  hw.output
}

hw.module @ShiftReg(in %a: i42, in %clk: !seq.clock, in %en: i1) {
  // CHECK: seq.shiftreg[2] sym @"shift/reg"
  %2 = seq.shiftreg[2] sym @"shift/reg" %a, %clk, %en : i42
  hw.output
}

hw.module @ClockGate(in %clk: !seq.clock, in %en: i1) {
  // CHECK: seq.clock_gate {{%.+}}, {{%.+}} sym @"gate name"
  %3 = seq.clock_gate %clk, %en sym @"gate name"
  hw.output
}
