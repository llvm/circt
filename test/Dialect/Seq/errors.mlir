// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @fifo1(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1) {
  // expected-error @+1 {{operation defines 3 results but was provided 5 to bind}}
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}

// -----

hw.module @fifo2(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1) {
  // expected-error @+1 {{'seq.fifo' op almost full threshold must be <= FIFO depth}}
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 almost_full 4 almost_empty 1 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}

// -----

hw.module @fifo3(in %clk : !seq.clock, in %rst : i1, in %in : i32, in %rdEn : i1, in %wrEn : i1) {
  // expected-error @+1 {{'seq.fifo' op almost empty threshold must be <= FIFO depth}}
  %out, %full, %empty, %almostFull, %almostEmpty = seq.fifo depth 3 almost_full 1 almost_empty 4 in %in rdEn %rdEn wrEn %wrEn clk %clk rst %rst : i32
}

// -----

hw.module @init() {
  // expected-error @+1 {{result type doesn't match with the terminator}}
  %0 = seq.initial {
    %1 = hw.constant 32: i32
    seq.yield %1, %1: i32, i32
  }: !seq.immutable<i32>
}

// -----

hw.module @init() {
  // expected-error @+1 {{'i32' is expected but got 'i16'}}
  %0 = seq.initial {
    %1 = hw.constant 32: i16
    seq.yield %1: i16
  }: !seq.immutable<i32>
}
