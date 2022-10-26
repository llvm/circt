// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @fifo2(%clk : i1, %rst : i1, %in : i32, %rdEn : i1, %wrEn : i1) -> () {
  // expected-error @+1 {{'seq.fifo' op combined output width must match input width (expected 32 but got 24)}}
  %out0, %out1, %empty, %full, %almostEmpty, %almostFull = seq.fifo[3] (%in, %rdEn, %wrEn) %clk, %rst : (i32) -> (i16, i8)
}
