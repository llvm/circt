// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-synth --convert-synth-to-comb -o %t.mlir
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

// RUN: circt-lec %t.mlir %s -c1=parity -c2=parity --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_PARITY
// COMB_PARITY: c1 == c2
hw.module @parity(in %arg0: i4, out out: i1) {
  %0 = comb.parity %arg0 : i4
  hw.output %0 : i1
}

// RUN: circt-lec %t.mlir %s -c1=add -c2=add --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ADD
// COMB_ADD: c1 == c2
hw.module @add(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 : i4
  hw.output %0 : i4
}

// RUN: circt-lec %t.mlir %s -c1=add_ripple_carry -c2=add_ripple_carry --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ADD_RIPPLE_CARRY
// COMB_ADD_RIPPLE_CARRY: c1 == c2
hw.module @add_ripple_carry(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.arch = "RIPPLE-CARRY"} : i4
  hw.output %0 : i4
}

// RUN: circt-lec %t.mlir %s -c1=add_sklanskey -c2=add_sklanskey --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ADD_SKLANSKEY
// COMB_ADD_SKLANSKEY: c1 == c2
hw.module @add_sklanskey(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.arch = "SKLANSKEY"} : i4
  hw.output %0 : i4
}

// RUN: circt-lec %t.mlir %s -c1=add_kogge_stone -c2=add_kogge_stone --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ADD_KOGGE_STONE
// COMB_ADD_KOGGE_STONE: c1 == c2
hw.module @add_kogge_stone(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.arch = "KOGGE-STONE"} : i4
  hw.output %0 : i4
}

// RUN: circt-lec %t.mlir %s -c1=add_brent_kung -c2=add_brent_kung --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ADD_BRENT_KUNG
// COMB_ADD_BRENT_KUNG: c1 == c2
hw.module @add_brent_kung(in %arg0: i4, in %arg1: i4, in %arg2: i4,  out add: i4) {
  %0 = comb.add %arg0, %arg1, %arg2 {synth.arch = "BRENT-KUNG"} : i4
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

// RUN: circt-lec %t.mlir %s -c1=icmp_eq_ne -c2=icmp_eq_ne --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_EQ_NE
// COMB_ICMP_EQ_NE: c1 == c2
hw.module @icmp_eq_ne(in %lhs: i3, in %rhs: i3, out out_eq: i1, out out_ne: i1) {
  %eq = comb.icmp eq %lhs, %rhs : i3
  %ne = comb.icmp ne %lhs, %rhs : i3
  hw.output %eq, %ne : i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=icmp_unsigned_compare -c2=icmp_unsigned_compare --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_UNSIGNED_COMPARE
// COMB_ICMP_UNSIGNED_COMPARE: c1 == c2
hw.module @icmp_unsigned_compare(in %lhs: i3, in %rhs: i3, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs : i3
  %uge = comb.icmp uge %lhs, %rhs : i3
  %ult = comb.icmp ult %lhs, %rhs : i3
  %ule = comb.icmp ule %lhs, %rhs : i3
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=icmp_signed_compare -c2=icmp_signed_compare --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_SIGNED_COMPARE
// COMB_ICMP_SIGNED_COMPARE: c1 == c2
hw.module @icmp_signed_compare(in %lhs: i3, in %rhs: i3, out out_sgt: i1, out out_sge: i1, out out_slt: i1, out out_sle: i1) {
  %sgt = comb.icmp sgt %lhs, %rhs : i3
  %sge = comb.icmp sge %lhs, %rhs : i3
  %slt = comb.icmp slt %lhs, %rhs : i3
  %sle = comb.icmp sle %lhs, %rhs : i3
  hw.output %sgt, %sge, %slt, %sle : i1, i1, i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=shift5 -c2=shift5 --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_SHIFT
// COMB_SHIFT: c1 == c2
hw.module @shift5(in %lhs: i5, in %rhs: i5, out out_shl: i5, out out_shr: i5, out out_shrs: i5) {
  %0 = comb.shl %lhs, %rhs : i5
  %1 = comb.shru %lhs, %rhs : i5
  %2 = comb.shrs %lhs, %rhs : i5
  hw.output %0, %1, %2 : i5, i5, i5
}
