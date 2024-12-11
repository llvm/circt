// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --convert-comb-to-aig --convert-aig-to-comb -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=bit_logical -c2=bit_logical --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_BIT_LOGICAL
// COMB_BIT_LOGICAL: c1 == c2
hw.module @bit_logical(in %arg0: i32, in %arg1: i32, in %arg2: i32, in %arg3: i32,
                in %cond: i1, out out0: i32, out out1: i32, out out2: i32, out out3: i32) {
  %0 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  %1 = comb.and %arg0, %arg1, %arg2, %arg3 : i32
  %2 = comb.xor %arg0, %arg1, %arg2, %arg3 : i32
  %3 = comb.mux %cond, %arg0, %arg1 : i32

  hw.output %0, %1, %2, %3 : i32, i32, i32, i32
}
