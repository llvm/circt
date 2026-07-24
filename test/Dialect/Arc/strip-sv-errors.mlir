// RUN: circt-opt %s --arc-strip-sv --split-input-file --verify-diagnostics

hw.module @AsyncReg(in %clk : !seq.clock, in %rst : i1, in %arg0: i8) {
  %c0_i8 = hw.constant 0 : i8
  // expected-error @below {{only synchronous resets are currently supported}}
  %int_rtc_tick_value = seq.firreg %arg0 clock %clk reset async %rst, %c0_i8 {clockEdge = 0 : i32, resetPolarity = 0 : i32} : i8
}

// -----

hw.module @AsyncHasBeenReset(in %clock: i1, in %reset: i1, out z: i1) {
  // expected-error @below {{has async reset, but only sync resets supported}}
  %0 = verif.has_been_reset %clock, async %reset
  hw.output %0 : i1
}

// -----

// `seq.compreg` is implicitly posedge-clocked, so a negedge register cannot be
// converted without changing its clock edge.
hw.module @NegedgeReg(in %clk : !seq.clock, in %arg0: i8) {
  // expected-error @below {{only posedge clocks are currently supported}}
  %r = seq.firreg %arg0 clock %clk {clockEdge = 1 : i32} : i8
}

// -----

// `seq.compreg` is implicitly active-high-reset, so an active-low reset cannot
// be converted without changing the reset polarity.
hw.module @ActiveLowReset(in %clk : !seq.clock, in %rst : i1, in %arg0: i8) {
  %c0_i8 = hw.constant 0 : i8
  // expected-error @below {{only active-high resets are currently supported}}
  %r = seq.firreg %arg0 clock %clk reset sync %rst, %c0_i8 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i8
}
