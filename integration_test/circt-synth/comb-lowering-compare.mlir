// REQUIRES: z3

// RUN: circt-opt %s --convert-comb-to-synth --convert-synth-to-comb -o %t.mlir

// RUN: circt-lec.sh %t.mlir %s -c1=icmp_unsigned_ripple_carry -c2=icmp_unsigned_ripple_carry
hw.module @icmp_unsigned_ripple_carry(in %lhs: i3, in %rhs: i3, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i3
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec.sh %t.mlir %s -c1=icmp_unsigned_sklanskey -c2=icmp_unsigned_sklanskey
hw.module @icmp_unsigned_sklanskey(in %lhs: i3, in %rhs: i3, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i3
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec.sh %t.mlir %s -c1=icmp_unsigned_kogge_stone -c2=icmp_unsigned_kogge_stone
// Use slightly larger width to verify the lazy prefix tree logic
hw.module @icmp_unsigned_kogge_stone(in %lhs: i14, in %rhs: i14, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i14
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i14
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i14
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i14
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec.sh %t.mlir %s -c1=icmp_unsigned_brent_kung -c2=icmp_unsigned_brent_kung
hw.module @icmp_unsigned_brent_kung(in %lhs: i4, in %rhs: i4, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  %uge = comb.icmp uge %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  %ult = comb.icmp ult %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  %ule = comb.icmp ule %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec.sh %t.mlir %s -c1=icmp_signed -c2=icmp_signed
hw.module @icmp_signed(in %lhs: i4, in %rhs: i4, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  // Sign comparisons are just unsigned comparisons with inverted inputs and outputs.
  // No need to test all architectures here.
  %ugt = comb.icmp sgt %lhs, %rhs {synth.test.arch = "RIPPLE-CARRY"} : i4
  %uge = comb.icmp sge %lhs, %rhs {synth.test.arch = "SKLANSKEY"} : i4
  %ult = comb.icmp slt %lhs, %rhs {synth.test.arch = "KOGGE-STONE"} : i4
  %ule = comb.icmp sle %lhs, %rhs {synth.test.arch = "BRENT-KUNG"} : i4
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// RUN: circt-lec.sh %t.mlir %s -c1=icmp_eq_ne -c2=icmp_eq_ne
hw.module @icmp_eq_ne(in %lhs: i3, in %rhs: i3, out out_eq: i1, out out_ne: i1) {
  %eq = comb.icmp eq %lhs, %rhs : i3
  %ne = comb.icmp ne %lhs, %rhs : i3
  hw.output %eq, %ne : i1, i1
}
