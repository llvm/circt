// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @err(%a: i1, %b: i1) -> () {
  // expected-error @+1 {{Expected lookup table of 2^n length}}
  %0 = comb.truth_table %a, %b -> [true, false]
}
