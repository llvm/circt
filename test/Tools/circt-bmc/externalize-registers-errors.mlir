// RUN: circt-opt --externalize-registers --split-input-file --verify-diagnostics %s

hw.module @two_clks(in %clk0: !seq.clock, in %clk1: !seq.clock, in %in: i32, out out: i32) {
  %1 = seq.compreg %in, %clk0 : i32
  // expected-warning @below {{multiple clocks not yet supported - all registers will be assumed to be clocked together}}
  %2 = seq.compreg %1, %clk1 : i32
  hw.output %2 : i32
}
