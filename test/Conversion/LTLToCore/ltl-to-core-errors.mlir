// RUN: circt-opt --lower-ltl-to-core --verify-diagnostics --split-input-file %s

hw.module @past_impl_clk(in %a: i32) {
  // expected-error @below {{ltl.past operations without a clock operand are not supported.}}
  %past = ltl.past %a, 2 : i32
}
