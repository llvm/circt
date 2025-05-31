// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-aig --convert-aig-to-comb -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=mul -c2=mul --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL
// COMB_MUL: c1 == c2
hw.module @mul(in %arg0: i7, in %arg1: i7, out add: i7) {
  %0 = comb.mul %arg0, %arg1 : i7
  hw.output %0 : i7
}
