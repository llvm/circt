// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-synth --convert-synth-to-comb -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=mul -c2=mul --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL
// COMB_MUL: c1 == c2
hw.module @mul(in %arg0: i7, in %arg1: i7, out add: i7) {
  %0 = comb.mul %arg0, %arg1 : i7
  hw.output %0 : i7
}

// RUN: circt-lec %t.mlir %s -c1=mul3 -c2=mul3 --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL_3
// COMB_MUL_3: c1 == c2
hw.module @mul3(in %arg0: i3, in %arg1: i3, in %arg2: i3, out add: i3) {
  %0 = comb.mul %arg0, %arg1, %arg2 : i3
  hw.output %0 : i3
}
