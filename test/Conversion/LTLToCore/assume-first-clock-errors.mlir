// RUN: circt-opt --lower-ltl-to-core='assume-first-clock' --verify-diagnostics --split-input-file %s

// Make sure we don't hallucinate some clock argument where there isn't one
hw.module @past_impl_clk(in %a: i32) {
  // expected-error @below {{failed to legalize operation 'ltl.past' that was explicitly marked illegal}}
  %past = ltl.past %a, 2 : i32
}
