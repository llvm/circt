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
hw.module @err(in %a: i4, in %b: i7) {
  // expected-note @+1 {{prior use here}}
  %0 = comb.concat %a, %a : i4, i4
  // expected-error @+1 {{use of value '%0' expects different type than prior uses: 'i7' vs 'i8'}}
  %1 = comb.and %0, %b : i7
}
