// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(hw-aggregate-to-comb,convert-comb-to-synth{target-ir=mig},convert-synth-to-comb))' -o %t.mlir
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

// RUN: circt-opt %s --convert-comb-to-synth -o %t2.mlir
// RUN: circt-lec %t2.mlir %s -c1=maj_inv_inverted -c2=maj_inv_inverted --shared-libs=%libz3 | FileCheck %s --check-prefix=MIG_INV
// MIG_INV: c1 == c2
hw.module @maj_inv_inverted(in %x: i1, in %y: i1, in %z: i1,
                            out out000: i1, out out001: i1, out out010: i1,
                            out out011: i1, out out100: i1, out out101: i1,
                            out out110: i1, out out111: i1) {
  %0 = synth.mig.maj_inv %x, %y, %z : i1
  %1 = synth.mig.maj_inv %x, %y, not %z : i1
  %2 = synth.mig.maj_inv %x, not %y, %z : i1
  %3 = synth.mig.maj_inv %x, not %y, not %z : i1
  %4 = synth.mig.maj_inv not %x, %y, %z : i1
  %5 = synth.mig.maj_inv not %x, %y, not %z : i1
  %6 = synth.mig.maj_inv not %x, not %y, %z : i1
  %7 = synth.mig.maj_inv not %x, not %y, not %z : i1
  hw.output %0, %1, %2, %3, %4, %5, %6, %7 : i1, i1, i1, i1, i1, i1, i1, i1
}
