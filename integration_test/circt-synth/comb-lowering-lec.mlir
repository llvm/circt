// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-synth -o %t.mlir
// RUN: circt-lec %t.mlir %s -c1=bit_logical -c2=bit_logical --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_BIT_LOGICAL
// COMB_BIT_LOGICAL: c1 == c2
hw.module @bit_logical(in %arg0: i4, in %arg1: i4, in %arg2: i4, in %arg3: i4,
                in %cond: i1, out out0: i4, out out1: i4, out out2: i4, out out3: i4) {
  %0 = comb.or %arg0, %arg1, %arg2, %arg3 : i4
  %1 = comb.and %arg0, %arg1, %arg2, %arg3 : i4
  %2 = comb.xor %arg0, %arg1, %arg2, %arg3 : i4
  %3 = comb.mux %cond, %arg0, %arg1 : i4

  hw.output %0, %1, %2, %3 : i4, i4, i4, i4
}

// RUN: circt-lec %t.mlir %s -c1=parity -c2=parity --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_PARITY
// COMB_PARITY: c1 == c2
hw.module @parity(in %arg0: i4, out out: i1) {
  %0 = comb.parity %arg0 : i4
  hw.output %0 : i1
}

// RUN: circt-lec %t.mlir %s -c1=sub -c2=sub --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_SUB
// COMB_SUB: c1 == c2
hw.module @sub(in %lhs: i4, in %rhs: i4, out out: i4) {
  %0 = comb.sub %lhs, %rhs : i4
  hw.output %0 : i4
}


// RUN: circt-lec %t.mlir %s -c1=shift5 -c2=shift5 --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_SHIFT
// COMB_SHIFT: c1 == c2
hw.module @shift5(in %lhs: i5, in %rhs: i5, out out_shl: i5, out out_shr: i5, out out_shrs: i5) {
  %0 = comb.shl %lhs, %rhs : i5
  %1 = comb.shru %lhs, %rhs : i5
  %2 = comb.shrs %lhs, %rhs : i5
  hw.output %0, %1, %2 : i5, i5, i5
}
