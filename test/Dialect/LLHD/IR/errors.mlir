// RUN: circt-opt %s --split-input-file --verify-diagnostics

hw.module @errors(in %in0: i32, out out0: i8) {
  // expected-error @below {{requires the same type for all operands and results}}
  %0 = "llhd.delay"(%in0) {delay = #llhd.time<0ns, 1d, 0e>} : (i32) -> i8
  hw.output %0 : i8
}
