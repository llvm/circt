// RUN: circt-lec %s --c1 divu_const_7 --c2 divu_const_7_lowered --emit-mlir | FileCheck %s --check-prefix=DIVU7
// RUN: circt-lec %s --c1 modu_const_3 --c2 modu_const_3_lowered --emit-mlir | FileCheck %s --check-prefix=MODU3
// RUN: circt-lec %s --c1 divs_const_neg3 --c2 divs_const_neg3_lowered --emit-mlir | FileCheck %s --check-prefix=DIVSNEG3
// RUN: circt-lec %s --c1 mods_const_neg3 --c2 mods_const_neg3_lowered --emit-mlir | FileCheck %s --check-prefix=MODSNEG3

// DIVU7-LABEL: smt.solver()
// DIVU7: smt.bv.udiv
// DIVU7: smt.bv.mul
// DIVU7: smt.bv.add
// DIVU7: smt.distinct

// MODU3-LABEL: smt.solver()
// MODU3: smt.bv.urem
// MODU3: smt.bv.mul
// MODU3: smt.bv.neg
// MODU3: smt.bv.add
// MODU3: smt.distinct

// DIVSNEG3-LABEL: smt.solver()
// DIVSNEG3: smt.bv.sdiv
// DIVSNEG3: smt.bv.mul
// DIVSNEG3: smt.bv.add
// DIVSNEG3: smt.distinct

// MODSNEG3-LABEL: smt.solver()
// MODSNEG3: smt.bv.srem
// MODSNEG3: smt.bv.mul
// MODSNEG3: smt.bv.add
// MODSNEG3: smt.distinct

hw.module @divu_const_7(in %lhs: i4, out out: i4) {
  %rhs = hw.constant 7 : i4
  %div = comb.divu %lhs, %rhs : i4
  hw.output %div : i4
}

hw.module @divu_const_7_lowered(in %lhs: i4, out out: i4) {
  %c0_i4 = hw.constant 0 : i4
  %wide = comb.concat %c0_i4, %lhs : i4, i4
  %magic = hw.constant 3 : i8
  %product = comb.mul %wide, %magic : i8
  %q0 = comb.extract %product from 4 : (i8) -> i4
  %diff = comb.sub %lhs, %q0 : i4
  %diff_bits = comb.extract %diff from 1 : (i4) -> i3
  %false = hw.constant false
  %diff_shifted = comb.concat %false, %diff_bits : i1, i3
  %q1 = comb.add %q0, %diff_shifted : i4
  %q_bits = comb.extract %q1 from 1 : (i4) -> i3
  %q = comb.concat %false, %q_bits : i1, i3
  hw.output %q : i4
}

hw.module @modu_const_3(in %lhs: i4, out out: i4) {
  %rhs = hw.constant 3 : i4
  %mod = comb.modu %lhs, %rhs : i4
  hw.output %mod : i4
}

hw.module @modu_const_3_lowered(in %lhs: i4, out out: i4) {
  %rhs = hw.constant 3 : i4
  %c0_i4 = hw.constant 0 : i4
  %wide = comb.concat %c0_i4, %lhs : i4, i4
  %magic = hw.constant 11 : i8
  %product = comb.mul %wide, %magic : i8
  %q0 = comb.extract %product from 4 : (i8) -> i4
  %q_bits = comb.extract %q0 from 1 : (i4) -> i3
  %false = hw.constant false
  %q = comb.concat %false, %q_bits : i1, i3
  %mul = comb.mul %q, %rhs : i4
  %mod = comb.sub %lhs, %mul : i4
  hw.output %mod : i4
}

hw.module @divs_const_neg3(in %lhs: i4, out out: i4) {
  %rhs = hw.constant -3 : i4
  %div = comb.divs %lhs, %rhs : i4
  hw.output %div : i4
}

hw.module @divs_const_neg3_lowered(in %lhs: i4, out out: i4) {
  %sign = comb.extract %lhs from 3 : (i4) -> i1
  %sign_ext = comb.replicate %sign : (i1) -> i4
  %wide = comb.concat %sign_ext, %lhs : i4, i4
  %magic = hw.constant 5 : i8
  %product = comb.mul %wide, %magic : i8
  %mulhi = comb.extract %product from 4 : (i8) -> i4
  %adjusted = comb.sub %mulhi, %lhs : i4
  %shift_sign = comb.extract %adjusted from 3 : (i4) -> i1
  %shift_bits = comb.extract %adjusted from 1 : (i4) -> i2
  %shift_pad = comb.replicate %shift_sign : (i1) -> i2
  %shifted = comb.concat %shift_pad, %shift_bits : i2, i2
  %result_sign = comb.extract %shifted from 3 : (i4) -> i1
  %c0_i3 = hw.constant 0 : i3
  %sign_padded = comb.concat %c0_i3, %result_sign : i3, i1
  %result = comb.add %shifted, %sign_padded : i4
  hw.output %result : i4
}

hw.module @mods_const_neg3(in %lhs: i4, out out: i4) {
  %rhs = hw.constant -3 : i4
  %mod = comb.mods %lhs, %rhs : i4
  hw.output %mod : i4
}

hw.module @mods_const_neg3_lowered(in %lhs: i4, out out: i4) {
  %rhs = hw.constant -3 : i4
  %sign = comb.extract %lhs from 3 : (i4) -> i1
  %sign_ext = comb.replicate %sign : (i1) -> i4
  %wide = comb.concat %sign_ext, %lhs : i4, i4
  %magic = hw.constant 5 : i8
  %product = comb.mul %wide, %magic : i8
  %mulhi = comb.extract %product from 4 : (i8) -> i4
  %adjusted = comb.sub %mulhi, %lhs : i4
  %shift_sign = comb.extract %adjusted from 3 : (i4) -> i1
  %shift_bits = comb.extract %adjusted from 1 : (i4) -> i2
  %shift_pad = comb.replicate %shift_sign : (i1) -> i2
  %shifted = comb.concat %shift_pad, %shift_bits : i2, i2
  %result_sign = comb.extract %shifted from 3 : (i4) -> i1
  %c0_i3 = hw.constant 0 : i3
  %sign_padded = comb.concat %c0_i3, %result_sign : i3, i1
  %quotient = comb.add %shifted, %sign_padded : i4
  %mul = comb.mul %quotient, %rhs : i4
  %mod = comb.sub %lhs, %mul : i4
  hw.output %mod : i4
}
