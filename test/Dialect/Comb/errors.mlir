// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @err(in %a: i1, in %b: i1) {
  // expected-error @+1 {{Expected lookup table of 2^n length}}
  %0 = comb.truth_table %a, %b -> [true, false]
}

// -----

hw.module @err(in %a: i1, in %b: i1) {
  // expected-error @+1 {{Truth tables support a maximum of 63 inputs}}
  %0 = comb.truth_table %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b, %a, %b -> [false]
}

// -----

hw.module @err(in %a: i0) {
  // expected-error @+1 {{op replicate must produce integer multiple of operand}}
  %0 = comb.replicate %a : (i0) -> i16
}

// -----

// This constraint is not absolutely necessary, and is enforced only to help diagnose bugs.
// See https://github.com/llvm/circt/pull/6959#discussion_r1588142448
hw.module @err(in %a: i0) {
  // expected-error @+1 {{from bit too large for input}}
  %0 = comb.extract %a from 1000 : (i0) -> i0
}

// -----

// Checks extract verifier does not overflow.
hw.module @err(in %a: i16) {
  // expected-error @+1 {{from bit too large for input}}
  %0 = comb.extract %a from 0xFFFFFFFF : (i16) -> i8
}
