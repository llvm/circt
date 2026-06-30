// RUN: circt-opt --arc-lower-processes --verify-diagnostics --split-input-file %s

// Observed-value clause on a wait: rejected.
hw.module @ObservedOnly(in %clk: i1) {
  llhd.process {
    // expected-error @below {{observed-value clauses are not supported}}
    llhd.wait (%clk : i1), ^bb1
  ^bb1:
    llhd.halt
  }
}

// -----

// Combined observed + delay: still rejected on the observed clause.
hw.module @ObservedAndDelay(in %clk: i1) {
  %t = llhd.constant_time <1ns, 0d, 0e>
  llhd.process {
    // expected-error @below {{observed-value clauses are not supported}}
    llhd.wait delay %t, (%clk : i1), ^bb1
  ^bb1:
    llhd.halt
  }
}
