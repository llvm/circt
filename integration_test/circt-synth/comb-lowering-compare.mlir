// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --convert-comb-to-synth --convert-synth-to-comb -o %t.mlir

// RUN: circt-lec %t.mlir %s -c1=icmp_unsigned_ripple_carry -c2=icmp_unsigned_ripple_carry --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_UNSIGNED_RIPPLE_CARRY
// COMB_ICMP_UNSIGNED_RIPPLE_CARRY: c1 == c2
hw.module @icmp_unsigned_ripple_carry(in %lhs: i3, in %rhs: i3, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=icmp_unsigned_sklanskey -c2=icmp_unsigned_sklanskey --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_UNSIGNED_SKLANSKEY
// COMB_ICMP_UNSIGNED_SKLANSKEY: c1 == c2
hw.module @icmp_unsigned_sklanskey(in %lhs: i3, in %rhs: i3, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=icmp_unsigned_kogge_stone -c2=icmp_unsigned_kogge_stone --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_UNSIGNED_KOGGE_STONE
// COMB_ICMP_UNSIGNED_KOGGE_STONE: c1 == c2
hw.module @icmp_unsigned_kogge_stone(in %lhs: i3, in %rhs: i3, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i3
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i3
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i3
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i3
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=icmp_unsigned_brent_kung -c2=icmp_unsigned_brent_kung --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_UNSIGNED_BRENT_KUNG
// COMB_ICMP_UNSIGNED_BRENT_KUNG: c1 == c2
hw.module @icmp_unsigned_brent_kung(in %lhs: i4, in %rhs: i4, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec %t.mlir %s -c1=icmp_signed -c2=icmp_signed --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMP_SIGNED
// COMB_ICMP_SIGNED: c1 == c2
hw.module @icmp_signed(in %lhs: i4, in %rhs: i4, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  // Sign comparisons are just unsigned comparisons with inverted inputs and outputs.
  // No need to test all architectures here.
  %ugt = comb.icmp sgt %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i4
  %uge = comb.icmp sge %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i4
  %ult = comb.icmp slt %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i4
  %ule = comb.icmp sle %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}
