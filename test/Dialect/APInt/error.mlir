// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @addSignless(%clk: i1, %rst: i1) {
  %c1_1 = arith.constant 1 : i1
  // expected-error @+1 {{'apint.add' op operand #0 must be an arbitrary precision integer with signedness semantics, but got 'i1'}}
  %0 = apint.add %c1_1, %c1_1 : i1, i1
}

// -----

hw.module @constantSignless(%clk: i1, %rst: i1) {
  // expected-error @+1 {{'apint.constant' op result #0 must be an arbitrary precision integer with signedness semantics, but got 'i1'}}
  %c1_1 = apint.constant 1 : i1
}
