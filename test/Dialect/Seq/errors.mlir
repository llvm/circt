// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @fifo1(input %clk : !seq.clock, input %rst : i1, input %in : i32, input %rdEn : i1, input %wrEn : i1) {
  // expected-error @+1 {{operation defines 3 results but was provided 5 to bind}}
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}

// -----

hw.module @fifo2(input %clk : !seq.clock, input %rst : i1, input %in : i32, input %rdEn : i1, input %wrEn : i1) {
  // expected-error @+1 {{'seq.fifo' op almost full threshold must be <= FIFO depth}}
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 almost_full 4 almost_empty 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}

// -----

hw.module @fifo3(input %clk : !seq.clock, input %rst : i1, input %in : i32, input %rdEn : i1, input %wrEn : i1) {
  // expected-error @+1 {{'seq.fifo' op almost empty threshold must be <= FIFO depth}}
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 almost_full 1 almost_empty 4 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}
