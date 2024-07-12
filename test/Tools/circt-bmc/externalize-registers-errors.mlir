// RUN: circt-opt --externalize-registers --split-input-file --verify-diagnostics %s

// expected-error @below {{modules with multiple clocks not yet supported}}
hw.module @two_clks(in %clk0: !seq.clock, in %clk1: !seq.clock, in %in: i32, out out: i32) {
  %1 = seq.compreg %in, %clk0 : i32
  %2 = seq.compreg %1, %clk1 : i32
  hw.output %2 : i32
}

// -----

hw.module @two_clks(in %clk_i1: i1, in %in: i32, out out: i32) {
  %clk = seq.to_clock %clk_i1
  // expected-error @below {{only clocks directly given as block arguments are supported}}
  %1 = seq.compreg %in, %clk : i32
  hw.output %1 : i32
}

// -----

hw.module @reg_with_reset(in %clk: !seq.clock, in %rst: i1, in %in: i32, out out: i32) {
  %c0_i32 = hw.constant 0 : i32
  // expected-error @below {{registers with reset signals not yet supported}}
  %1 = seq.compreg %in, %clk reset %rst, %c0_i32 : i32
  hw.output %1 : i32
}

// -----

hw.module @reg_with_poweron(in %clk: !seq.clock, in %in: i32, out out: i32) {
  %c0_i32 = hw.constant 0 : i32
  // expected-error @below {{registers with power-on values not yet supported}}
  %1 = seq.compreg %in, %clk powerOn %c0_i32 : i32
  hw.output %1 : i32
}
