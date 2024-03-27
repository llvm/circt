// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @NeedsBothResetAndResetValue(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op must specify reset and reset value}}
  "seq.firreg"(%input, %clk) { name = "reg", isAsync } : (i1, !seq.clock) -> i1
}

// -----

// CHECK-LABEL: @preset_too_large
hw.module @preset_too_large(in %clock: !seq.clock, in %reset: i1, in %next4: i4) {
  // expected-error@below {{custom op 'seq.firreg' preset value too large}}
  seq.firreg %next4 clock %clock preset 1024 : i4
}
