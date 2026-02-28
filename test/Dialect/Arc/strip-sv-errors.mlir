// RUN: circt-opt %s --arc-strip-sv --split-input-file --verify-diagnostics

hw.module @AsyncReg(in %clk : !seq.clock, in %rst : i1, in %arg0: i8) {
  %c0_i8 = hw.constant 0 : i8
  // expected-error @below {{only synchronous resets are currently supported}}
  %int_rtc_tick_value = seq.firreg %arg0 clock %clk reset async %rst, %c0_i8 : i8
}
