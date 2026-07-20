// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @NeedsBothResetAndResetValue(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op must specify reset and reset value}}
  "seq.firreg"(%input, %clk) { name = "reg", resetType = 1 : i32 } : (i1, !seq.clock) -> i1
}

// -----

hw.module @ResetPolarityWithoutReset(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op 'resetPolarity' is only valid on a register with a reset}}
  "seq.firreg"(%input, %clk) { name = "reg", resetPolarity = 1 : i32 } : (i1, !seq.clock) -> i1
}

// -----

hw.module @ResetTypeWithoutReset(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op 'resetType' is only valid on a register with a reset}}
  "seq.firreg"(%input, %clk) { name = "reg", resetType = 0 : i32 } : (i1, !seq.clock) -> i1
}

// -----

hw.module @LegacyIsAsyncAttr(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op has the legacy 'isAsync' attribute}}
  %r = seq.firreg %input clock %clk {isAsync} : i1
}

// -----

// CHECK-LABEL: @preset_too_large
hw.module @preset_too_large(in %clock: !seq.clock, in %reset: i1, in %next4: i4) {
  // expected-error@below {{custom op 'seq.firreg' preset value too large}}
  seq.firreg %next4 clock %clock preset 1024 : i4
}

// -----

// CHECK-LABEL: @preset_negative
hw.module @preset_negative(in %clock: !seq.clock, in %reset: i1, in %next: i1) {
  // expected-error@below {{custom op 'seq.firreg' preset value must not be negative}}
  seq.firreg %next clock %clock preset -1 : i1
}
