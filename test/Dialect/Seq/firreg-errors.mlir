// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @NeedsBothResetAndResetValue(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op must specify reset and reset value}}
  "seq.firreg"(%input, %clk) { name = "reg", resetType = 1 : i32, clockEdge = 0 : i32 } : (i1, !seq.clock) -> i1
}

// -----

hw.module @ResetPolarityWithoutReset(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op 'resetPolarity' is only valid on a register with a reset}}
  "seq.firreg"(%input, %clk) { name = "reg", resetPolarity = 1 : i32, clockEdge = 0 : i32 } : (i1, !seq.clock) -> i1
}

// -----

hw.module @ResetTypeWithoutReset(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op 'resetType' is only valid on a register with a reset}}
  "seq.firreg"(%input, %clk) { name = "reg", resetType = 0 : i32, clockEdge = 0 : i32 } : (i1, !seq.clock) -> i1
}

// -----

hw.module @LegacyIsAsyncAttr(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op has the legacy 'isAsync' attribute}}
  %r = seq.firreg %input clock %clk {clockEdge = 0 : i32, isAsync} : i1
}

// -----

// A dual-edge clock is not valid synthesizable logic.
hw.module @DualEdgeClock(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op has 'clockEdge = both' (dual-edge), which is not valid synthesizable logic; use 'pos' or 'neg'}}
  %r = seq.firreg %input clock %clk {clockEdge = 2 : i32} : i1
}

// -----

// clockEdge is required on every register.
hw.module @MissingClockEdge(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op requires attribute 'clockEdge'}}
  %r = seq.firreg %input clock %clk : i1
}

// -----

// A reset-bearing register requires an explicit resetPolarity.
hw.module @MissingResetPolarity(in %input: i1, in %clk: !seq.clock, in %rst: i1) {
  // expected-error@+1 {{'seq.firreg' op requires 'resetPolarity' on a register with a reset}}
  %r = seq.firreg %input clock %clk reset sync %rst, %input {clockEdge = 0 : i32} : i1
}

// -----

// A reset-bearing register requires an explicit resetType.
hw.module @MissingResetType(in %input: i1, in %clk: !seq.clock, in %rst: i1) {
  // expected-error@+1 {{'seq.firreg' op requires 'resetType' on a register with a reset}}
  "seq.firreg"(%input, %clk, %rst, %input) { name = "reg", clockEdge = 0 : i32, resetPolarity = 0 : i32 } : (i1, !seq.clock, i1, i1) -> i1
}

// -----

// CHECK-LABEL: @preset_too_large
hw.module @preset_too_large(in %clock: !seq.clock, in %reset: i1, in %next4: i4) {
  // expected-error@below {{custom op 'seq.firreg' preset value too large}}
  seq.firreg %next4 clock %clock preset 1024 {clockEdge = 0 : i32} : i4
}

// -----

// CHECK-LABEL: @preset_negative
hw.module @preset_negative(in %clock: !seq.clock, in %reset: i1, in %next: i1) {
  // expected-error@below {{custom op 'seq.firreg' preset value must not be negative}}
  seq.firreg %next clock %clock preset -1 {clockEdge = 0 : i32} : i1
}
