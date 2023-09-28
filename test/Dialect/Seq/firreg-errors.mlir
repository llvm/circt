// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @NeedsBothResetAndResetValue(in %input: i1, in %clk: !seq.clock) {
  // expected-error@+1 {{'seq.firreg' op must specify reset and reset value}}
  "seq.firreg"(%input, %clk) { name = "reg", isAsync } : (i1, !seq.clock) -> i1
}
