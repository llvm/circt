// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @err(in %a: i8, in %b: i8) {
  // expected-error @+1 {{'datapath.compress' op requires 3 or more arguments - otherwise use add}}
  %0:2 = datapath.compress %a, %b : i8 [2 -> 2]
}

// -----

hw.module @err(in %a: i8, in %b: i8, in %c: i8) {
  // expected-error @+1 {{'datapath.compress' op must produce at least 2 results}}
  %0 = datapath.compress %a, %b, %c : i8 [3 -> 1]
}

// -----

hw.module @err(in %a: i8, in %b: i8, in %c: i8) {
  // expected-error @+1 {{'datapath.compress' op must reduce the number of operands by at least 1}}
  %0:3 = datapath.compress %a, %b, %c : i8 [3 -> 3]
}

// -----

hw.module @err(in %a: i8, in %b: i8, in %c: i8) {
  // expected-error @+1 {{'datapath.compress' op must reduce the number of operands by at least 1}}
  %0:4 = datapath.compress %a, %b, %c : i8 [3 -> 4]
}
