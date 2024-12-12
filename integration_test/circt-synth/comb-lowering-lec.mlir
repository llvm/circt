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

// RUN: circt-lec %t.mlir %s -c1=add -c2=add --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ADD
// COMB_ADD: c1 == c2
hw.module @add(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 : i4
  hw.output %0 : i4
}

// RUN: circt-lec %t.mlir %s -c1=sub -c2=sub --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_SUB
// COMB_SUB: c1 == c2
hw.module @sub(in %lhs: i4, in %rhs: i4, out out: i4) {
  %0 = comb.sub %lhs, %rhs : i4
  hw.output %0 : i4
}

// RUN: circt-lec %t.mlir %s -c1=mul -c2=mul --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL
// COMB_MUL: c1 == c2
hw.module @mul(in %arg0: i3, in %arg1: i3, in %arg2: i3, out add: i3) {
  %0 = comb.mul %arg0, %arg1, %arg2 : i3
  hw.output %0 : i3
}
