// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @signlessOperands() {
  %c1_1 = arith.constant 1 : i1
  // expected-error @+1 {{'hwarith.add' op operand #0 must be an arbitrary precision integer with signedness semantics, but got 'i1'}}
  %0 = hwarith.add %c1_1, %c1_1 : (i1, i1) -> i2
}

// -----

hw.module @signlessResult() {
  %c1_1 = hwarith.constant 1 : ui1
  // expected-error @+1 {{'hwarith.add' op result #0 must be an arbitrary precision integer with signedness semantics, but got 'i2'}}
  %0 = hwarith.add %c1_1, %c1_1 : (ui1, ui1) -> i2
}

// -----

hw.module @wrongResultWidth() {
  %c1_1 = hwarith.constant 1 : ui1
  // expected-error @+1 {{'hwarith.add' op inferred type(s) 'ui2' are incompatible with return type(s) of operation 'ui3'}}
  %0 = hwarith.add %c1_1, %c1_1 : (ui1, ui1) -> ui3
}

// -----

hw.module @signlessConst() {
  // expected-error @+1 {{'hwarith.constant' op result #0 must be an arbitrary precision integer with signedness semantics, but got 'i1'}}
  %c1_1 = hwarith.constant 1 : i1
}

// -----

hw.module @zeroSizeConst() {
  // expected-error @+1 {{'hwarith.constant' op result #0 must be an arbitrary precision integer with signedness semantics, but got 'ui0'}}
  %c1_1 = hwarith.constant 0 : ui0
}
