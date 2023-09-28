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
