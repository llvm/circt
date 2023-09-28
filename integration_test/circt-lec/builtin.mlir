// These tests will be only enabled if circt-lec is built.
// REQUIRES: circt-lec

// builtin.module implementation
//  RUN: circt-lec %s -v=false | FileCheck %s --check-prefix=BUILTIN_MODULE
//  BUILTIN_MODULE: c1 == c2

hw.module @basic(in %in: i1, out out: i1) {
  hw.output %in : i1
}
