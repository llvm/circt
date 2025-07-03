// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @err(in %a: i8, in %b: i8) {
  // expected-error @+1 {{'datapath.compress' op requires 3 or more arguments - otherwise use add}}
  %0:2 = datapath.compress %a, %b : 2 x i8 -> (i8, i8)
}

// -----

hw.module @err(in %a: i8, in %b: i8, in %c: i8) {
  // expected-error @+1 {{'datapath.compress' op must produce at least 2 results}}
  %0 = datapath.compress %a, %b, %c : 3 x i8 -> (i8)
}

// -----

hw.module @err(in %a: i8, in %b: i8, in %c: i8) {
  // expected-error @+1 {{'datapath.compress' op must reduce the number of operands by at least 1}}
  %0:3 = datapath.compress %a, %b, %c : 3 x i8 -> (i8, i8, i8)
}

// -----

hw.module @err(in %a: i8, in %b: i8, in %c: i8) {
  // expected-error @+1 {{'datapath.compress' op must reduce the number of operands by at least 1}}
  %0:4 = datapath.compress %a, %b, %c : 3 x i8 -> (i8, i8, i8, i8)
}
