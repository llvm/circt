// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-synth --convert-synth-to-comb -o %t.mlir

// RUN: circt-lec %t.mlir %s -c1=divmodu -c2=divmodu --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVMODU
// COMB_DIVMODU: c1 == c2
hw.module @divmodu(in %lhs: i3, in %rhs: i3, out out_div: i3, out out_mod: i3) {
  %c0_i3 = hw.constant 0 : i3
  %neq = comb.icmp ne %rhs, %c0_i3 : i3
  verif.assume %neq : i1

  %0 = comb.divu %lhs, %rhs : i3
  %1 = comb.modu %lhs, %rhs : i3
  hw.output %0, %1 : i3, i3
}

// RUN: circt-lec %t.mlir %s -c1=divmodu_power_of_two -c2=divmodu_power_of_two --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVMODU_POWER_OF_TWO
// COMB_DIVMODU_POWER_OF_TWO: c1 == c2
hw.module @divmodu_power_of_two(in %lhs: i8, out out_div: i8, out out_mod: i8) {
  %c16_i8 = hw.constant 16 : i8

  %0 = comb.divu %lhs, %c16_i8 : i8
  %1 = comb.modu %lhs, %c16_i8 : i8
  hw.output %0, %1 : i8, i8
}

// RUN: circt-lec %t.mlir %s -c1=divmods -c2=divmods --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVMODS
// COMB_DIVMODS: c1 == c2
hw.module @divmods(in %lhs: i3, in %rhs: i3, out out_div: i3, out out_mod: i3) {
  %c0_i3 = hw.constant 0 : i3
  %neq = comb.icmp ne %rhs, %c0_i3 : i3
  verif.assume %neq : i1

  %0 = comb.divs %lhs, %rhs : i3
  %1 = comb.mods %lhs, %rhs : i3
  hw.output %0, %1 : i3, i3
}

// RUN: circt-lec %t.mlir %s -c1=divmod_mix_constant -c2=divmod_mix_constant --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVMOD_MIX_CONSTANT
// COMB_DIVMOD_MIX_CONSTANT: c1 == c2
hw.module @divmod_mix_constant(in %in: i1, in %lhs: i1, in %rhs: i1, out out_divu: i4, out out_modu: i4, out out_divs: i4, out out_mods: i4) {
  %c2_i2 = hw.constant 2 : i2

  %new_lhs = comb.concat %in, %c2_i2, %lhs : i1, i2, i1
  %new_rhs = comb.concat %c2_i2, %rhs, %in : i2, i1, i1
  %0 = comb.divu %new_lhs, %new_rhs : i4
  %1 = comb.modu %new_lhs, %new_rhs : i4
  %2 = comb.divs %new_lhs, %new_rhs : i4
  %3 = comb.mods %new_lhs, %new_rhs : i4
  hw.output %0, %1, %2, %3 : i4, i4, i4, i4
}

