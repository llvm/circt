// REQUIRES: z3

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-synth --convert-synth-to-comb -o %t.mlir

// Wide non-constant divisors are lowered to a restoring array divider.
// RUN: circt-lec.sh %t.mlir %s -c1=divmodu_dynamic -c2=divmodu_dynamic
hw.module @divmodu_dynamic(in %lhs: i8, in %rhs: i8, out out_div: i8, out out_mod: i8) {
  %c0_i8 = hw.constant 0 : i8
  %neq = comb.icmp ne %rhs, %c0_i8 : i8
  verif.assume %neq : i1

  %0 = comb.divu %lhs, %rhs : i8
  %1 = comb.modu %lhs, %rhs : i8
  hw.output %0, %1 : i8, i8
}

// RUN: circt-lec.sh %t.mlir %s -c1=divmods_dynamic -c2=divmods_dynamic
hw.module @divmods_dynamic(in %lhs: i8, in %rhs: i8, out out_div: i8, out out_mod: i8) {
  %c0_i8 = hw.constant 0 : i8
  %neq = comb.icmp ne %rhs, %c0_i8 : i8
  verif.assume %neq : i1

  %0 = comb.divs %lhs, %rhs : i8
  %1 = comb.mods %lhs, %rhs : i8
  hw.output %0, %1 : i8, i8
}
