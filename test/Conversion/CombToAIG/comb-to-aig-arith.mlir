// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-comb-to-aig{additional-legal-ops=comb.xor,comb.or,comb.and,comb.mux},cse))" | FileCheck %s
// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-comb-to-aig{additional-legal-ops=comb.xor,comb.or,comb.and,comb.mux,comb.add},cse))" | FileCheck %s --check-prefix=ALLOW_ADD
// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-comb-to-aig{additional-legal-ops=comb.xor,comb.or,comb.and,comb.mux,comb.icmp}))" | FileCheck %s --check-prefix=ALLOW_ICMP

// CHECK-LABEL: @parity
hw.module @parity(in %arg0: i4, out out: i1) {
  // CHECK-NEXT: %[[ext0:.+]] = comb.extract %arg0 from 0 : (i4) -> i1
  // CHECK-NEXT: %[[ext1:.+]] = comb.extract %arg0 from 1 : (i4) -> i1
  // CHECK-NEXT: %[[ext2:.+]] = comb.extract %arg0 from 2 : (i4) -> i1
  // CHECK-NEXT: %[[ext3:.+]] = comb.extract %arg0 from 3 : (i4) -> i1
  // CHECK-NEXT: %[[xor:.+]] = comb.xor bin %[[ext0]], %[[ext1]], %[[ext2]], %[[ext3]] : i1
  // CHECK-NEXT: hw.output %[[xor]] : i1
  %0 = comb.parity %arg0 : i4
  hw.output %0 : i1
}

// CHECK-LABEL: @add
hw.module @add(in %lhs: i2, in %rhs: i2, out out: i2) {
  // CHECK:      %[[lhs0:.*]] = comb.extract %lhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[lhs1:.*]] = comb.extract %lhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[rhs0:.*]] = comb.extract %rhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[rhs1:.*]] = comb.extract %rhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[sum0:.*]] = comb.xor bin %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[carry0:.*]] = comb.and bin %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[sum1:.*]] = comb.xor bin %[[lhs1]], %[[rhs1]], %[[carry0]] : i1
  // CHECK-NEXT: %[[concat:.*]] = comb.concat %[[sum1]], %[[sum0]] : i1, i1
  // CHECK-NEXT: hw.output %[[concat]] : i2
  %0 = comb.add %lhs, %rhs : i2
  hw.output %0 : i2
}

// CHECK-LABEL: @add_17
hw.module @add_17(in %lhs: i17, in %rhs: i17, out out: i17) {
  %0 = comb.add %lhs, %rhs : i17
  hw.output %0 : i17
}

// CHECK-LABEL: @add_35
hw.module @add_35(in %lhs: i35, in %rhs: i35, out out: i35) {
  %0 = comb.add %lhs, %rhs : i35
  hw.output %0 : i35
}

// CHECK-LABEL: @add_64
hw.module @add_64(in %lhs: i64, in %rhs: i64, out out: i64) {
  %0 = comb.add %lhs, %rhs : i64
  hw.output %0 : i64
}

// CHECK-LABEL: @sub
// ALLOW_ADD-LABEL: @sub
// ALLOW_ADD-NEXT: %[[NOT_RHS:.+]] = aig.and_inv not %rhs
// ALLOW_ADD-NEXT: %[[CONST:.+]] = hw.constant 1 : i4
// ALLOW_ADD-NEXT: %[[ADD:.+]] = comb.add bin %lhs, %[[NOT_RHS]], %[[CONST]] {sv.namehint = "sub"}
// ALLOW_ADD-NEXT: hw.output %[[ADD]]
hw.module @sub(in %lhs: i4, in %rhs: i4, out out: i4) {
  %0 = comb.sub %lhs, %rhs {sv.namehint = "sub"} : i4
  hw.output %0 : i4
}


// CHECK-LABEL: @mul
// ALLOW_ADD-LABEL:  hw.module @mul(in %lhs : i3, in %rhs : i3, out out : i3) {
// ALLOW_ADD-NEXT: %[[LHS0:.+]] = comb.extract %lhs from 0 : (i3) -> i1
// ALLOW_ADD-NEXT: %[[LHS1:.+]] = comb.extract %lhs from 1 : (i3) -> i1
// ALLOW_ADD-NEXT: %[[LHS2:.+]] = comb.extract %lhs from 2 : (i3) -> i1
// ALLOW_ADD-NEXT: %[[RHS0:.+]] = comb.extract %rhs from 0 : (i3) -> i1
// ALLOW_ADD-NEXT: %[[RHS1:.+]] = comb.extract %rhs from 1 : (i3) -> i1
// ALLOW_ADD-NEXT: %[[RHS2:.+]] = comb.extract %rhs from 2 : (i3) -> i1
//                  Partial Products
// ALLOW_ADD-NEXT: %[[P_0_0:.+]] = comb.and %[[LHS0]], %[[RHS0]] : i1
// ALLOW_ADD-NEXT: %[[P_1_0:.+]] = comb.and %[[LHS1]], %[[RHS0]] : i1
// ALLOW_ADD-NEXT: %[[P_2_0:.+]] = comb.and %[[LHS2]], %[[RHS0]] : i1
// ALLOW_ADD-NEXT: %[[P_0_1:.+]] = comb.and %[[LHS0]], %[[RHS1]] : i1
// ALLOW_ADD-NEXT: %[[P_1_1:.+]] = comb.and %[[LHS1]], %[[RHS1]] : i1
// ALLOW_ADD-NEXT: %[[P_0_2:.+]] = comb.and %[[LHS0]], %[[RHS2]] : i1
//                    Wallace Tree Reduction
// ALLOW_ADD-NEXT: %[[XOR0:.+]] = comb.xor bin %[[P_0_2]], %[[P_1_1]] : i1
// ALLOW_ADD-NEXT: %[[XOR1:.+]] = comb.xor bin %[[XOR0]], %[[P_2_0]] : i1
// ALLOW_ADD-NEXT: %[[AND0:.+]] = comb.and bin %[[P_0_2]], %[[P_1_1]] : i1
// ALLOW_ADD-NEXT: %[[AND1:.+]] = comb.and bin %[[XOR0]], %[[P_2_0]] : i1
// ALLOW_ADD-NEXT: %false = hw.constant false
// ALLOW_ADD-NEXT: %[[SUM_ROW:.+]] = comb.concat %[[XOR1]], %[[P_1_0]], %[[P_0_0]] : i1, i1, i1
// ALLOW_ADD-NEXT: %[[CARRY_ROW:.+]] = comb.concat %false, %[[P_0_1]], %false : i1, i1, i1
// ALLOW_ADD-NEXT: comb.add bin %[[SUM_ROW]], %[[CARRY_ROW]] : i3
hw.module @mul(in %lhs: i3, in %rhs: i3, out out: i3) {
  %0 = comb.mul %lhs, %rhs : i3
  hw.output %0 : i3
}

// CHECK-LABEL: @mul_0
// CHECK-NEXT: %[[C0:.+]] = hw.constant 0 : i0
// CHECK-NEXT: hw.output %[[C0]] : i0
hw.module @mul_0(in %lhs: i0, in %rhs: i0, out out: i0) {
  %0 = comb.mul %lhs, %rhs : i0
  hw.output %0 : i0
}

// CHECK-LABEL: @mul_1
// CHECK-NEXT: %[[AND:.+]] = comb.and %lhs, %rhs : i1
// CHECK-NEXT: hw.output %[[AND]] : i1
hw.module @mul_1(in %lhs: i1, in %rhs: i1, out out: i1) {
  %0 = comb.mul %lhs, %rhs : i1
  hw.output %0 : i1
}

// CHECK-LABEL: @mul_17
hw.module @mul_17(in %lhs: i17, in %rhs: i17, out out: i17) {
  %0 = comb.mul %lhs, %rhs : i17
  hw.output %0 : i17
}

// CHECK-LABEL: @icmp_eq_ne
hw.module @icmp_eq_ne(in %lhs: i2, in %rhs: i2, out out_eq: i1, out out_ne: i1) {
  %eq = comb.icmp eq %lhs, %rhs : i2
  %ne = comb.icmp ne %lhs, %rhs : i2
  // CHECK-NEXT:   %[[XOR:.+]] = comb.xor %lhs, %rhs
  // CHECK-NEXT:   %[[XOR_0:.+]] = comb.extract %[[XOR]] from 0 : (i2) -> i1
  // CHECK-NEXT:   %[[XOR_1:.+]] = comb.extract %[[XOR]] from 1 : (i2) -> i1
  // CHECK-NEXT:   %[[EQ:.+]] = aig.and_inv not %[[XOR_0]], not %[[XOR_1]]
  // CHECK-NEXT:   %[[NEQ:.+]] = comb.or bin %[[XOR_0]], %[[XOR_1]]
  // CHECK-NEXT:   hw.output %[[EQ]], %[[NEQ]]
  // CHECK-NEXT: }
  hw.output %eq, %ne : i1, i1
}

// CHECK-LABEL: @icmp_unsigned_compare
hw.module @icmp_unsigned_compare(in %lhs: i2, in %rhs: i2, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp ugt %lhs, %rhs : i2
  %uge = comb.icmp uge %lhs, %rhs : i2
  %ult = comb.icmp ult %lhs, %rhs : i2
  %ule = comb.icmp ule %lhs, %rhs : i2
  // CHECK-NEXT:   %[[LHS_0:.+]] = comb.extract %lhs from 0 : (i2) -> i1
  // CHECK-NEXT:   %[[LHS_1:.+]] = comb.extract %lhs from 1 : (i2) -> i1
  // CHECK-NEXT:   %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // CHECK-NEXT:   %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // CHECK-NEXT:   %[[LSB_NEQ:.+]] = comb.xor bin %[[LHS_0]], %[[RHS_0]]
  // CHECK-NEXT:   %[[LSB_GT:.+]] = aig.and_inv %[[LHS_0]], not %[[RHS_0]]
  // CHECK-NEXT:   %[[MSB_NEQ:.+]] = comb.xor bin %[[LHS_1]], %[[RHS_1]]
  // CHECK-NEXT:   %[[MSB_EQ:.+]] = aig.and_inv not %[[MSB_NEQ]]
  // CHECK-NEXT:   %[[MSB_GT:.+]] = aig.and_inv %[[LHS_1]], not %[[RHS_1]]
  // CHECK-NEXT:   %[[MSB_EQ_AND_LSB_GT:.+]] = comb.and bin %[[MSB_EQ]], %[[LSB_GT]]
  // CHECK-NEXT:   %[[UGT:.+]] = comb.or bin %[[MSB_GT]], %[[MSB_EQ_AND_LSB_GT]]
  // CHECK-NEXT:   %[[LSB_EQ:.+]] = aig.and_inv not %[[LSB_NEQ]]
  // CHECK-NEXT:   %[[LSB_UGE:.+]] = comb.or bin %[[LSB_GT]], %[[LSB_EQ]]
  // CHECK-NEXT:   %[[MSB_EQ_AND_LSB_UGE:.+]] = comb.and bin %[[MSB_EQ]], %[[LSB_UGE]]
  // CHECK-NEXT:   %[[UGE:.+]] = comb.or bin %[[MSB_GT]], %[[MSB_EQ_AND_LSB_UGE]]
  // CHECK-NEXT:   %[[LSB_LT:.+]] = aig.and_inv not %[[LHS_0]], %[[RHS_0]]
  // CHECK-NEXT:   %[[MSB_LT:.+]] = aig.and_inv not %[[LHS_1]], %[[RHS_1]]
  // CHECK-NEXT:   %[[MSB_EQ_AND_LSB_LT:.+]] = comb.and bin %[[MSB_EQ]], %[[LSB_LT]]
  // CHECK-NEXT:   %[[ULT:.+]] = comb.or bin %[[MSB_LT]], %[[MSB_EQ_AND_LSB_LT]]
  // CHECK-NEXT:   %[[LSB_LE:.+]] = comb.or bin %[[LSB_LT]], %[[LSB_EQ]]
  // CHECK-NEXT:   %[[MSB_EQ_AND_LSB_LE:.+]] = comb.and bin %[[MSB_EQ]], %[[LSB_LE]]
  // CHECK-NEXT:   %[[ULE:.+]] = comb.or bin %[[MSB_LT]], %[[MSB_EQ_AND_LSB_LE]]
  // CHECK-NEXT:   hw.output %[[UGT]], %[[UGE]], %[[ULT]], %[[ULE]]
  // CHECK-NEXT: }
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}

// CHECK-LABEL: @icmp_signed_compare
hw.module @icmp_signed_compare(in %lhs: i2, in %rhs: i2, out out_sgt: i1, out out_sge: i1, out out_slt: i1, out out_sle: i1) {
  %sgt = comb.icmp sgt %lhs, %rhs : i2
  %sge = comb.icmp sge %lhs, %rhs : i2
  %slt = comb.icmp slt %lhs, %rhs : i2
  %sle = comb.icmp sle %lhs, %rhs : i2
  // CHECK-NEXT:   %[[LHS_0:.+]] = comb.extract %lhs from 0 : (i2) -> i1
  // CHECK-NEXT:   %[[LHS_1:.+]] = comb.extract %lhs from 1 : (i2) -> i1
  // CHECK-NEXT:   %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // CHECK-NEXT:   %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // CHECK-NEXT:   %[[LSB_NEQ:.+]] = comb.xor bin %[[LHS_0]], %[[RHS_0]]
  // CHECK-NEXT:   %[[LSB_GT:.+]] = aig.and_inv %[[LHS_0]], not %[[RHS_0]]
  // CHECK-NEXT:   %[[SIGN_NEQ:.+]] = comb.xor %[[LHS_1]], %[[RHS_1]]
  // CHECK-NEXT:   %[[SGT:.+]] = comb.mux %[[SIGN_NEQ]], %[[RHS_1]], %[[LSB_GT]]
  // CHECK-NEXT:   %[[LSB_EQ:.+]] = aig.and_inv not %[[LSB_NEQ]]
  // CHECK-NEXT:   %[[LSB_GE:.+]] = comb.or bin %[[LSB_GT]], %[[LSB_EQ]]
  // CHECK-NEXT:   %[[SGE:.+]] = comb.mux %[[SIGN_NEQ]], %[[RHS_1]], %[[LSB_GE]]
  // CHECK-NEXT:   %[[LSB_LT:.+]] = aig.and_inv not %[[LHS_0]], %[[RHS_0]]
  // CHECK-NEXT:   %[[SLT:.+]] = comb.mux %[[SIGN_NEQ]], %[[LHS_1]], %[[LSB_LT]]
  // CHECK-NEXT:   %[[LSB_LE:.+]] = comb.or bin %[[LSB_LT]], %[[LSB_EQ]]
  // CHECK-NEXT:   %[[SLE:.+]] = comb.mux %[[SIGN_NEQ]], %[[LHS_1]], %[[LSB_LE]]
  // CHECK-NEXT:   hw.output %[[SGT]], %[[SGE]], %[[SLT]], %[[SLE]]
  // CHECK-NEXT: }
  hw.output %sgt, %sge, %slt, %sle : i1, i1, i1, i1
}

// CHECK-LABEL: @shift2
// ALLOW_ICMP-LABEL: @shift2
hw.module @shift2(in %lhs: i2, in %rhs: i2, out out_shl: i2, out out_shr: i2, out out_shrs: i2) {
  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[LHS_0:.+]] = comb.extract %lhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[FALSE:.+]] = hw.constant false
  // ALLOW_ICMP-NEXT: %[[L_SHIFT_BY_1:.+]] = comb.concat %[[LHS_0]], %[[FALSE]]
  // ALLOW_ICMP-NEXT: %[[C0_I2:.+]] = hw.constant 0
  // ALLOW_ICMP-NEXT: %[[L_SHIFT:.+]] = comb.mux %[[RHS_0]], %[[L_SHIFT_BY_1]], %lhs
  // ALLOW_ICMP-NEXT: %[[C3_I2:.+]] = hw.constant -2
  // ALLOW_ICMP-NEXT: %[[ICMP:.+]] = comb.icmp ult %rhs, %[[C3_I2]]
  // ALLOW_ICMP-NEXT: %[[L_SHIFT_WITH_BOUND_CHECK:.+]] = comb.mux %[[ICMP]], %[[L_SHIFT]], %[[C0_I2]]
  %0 = comb.shl %lhs, %rhs : i2

  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[LHS_1:.+]] = comb.extract %lhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[FALSE:.+]] = hw.constant false
  // ALLOW_ICMP-NEXT: %[[R_SHIFT_BY_1:.+]] = comb.concat %[[FALSE]], %[[LHS_1]]
  // ALLOW_ICMP-NEXT: %[[C0_I2:.+]] = hw.constant 0
  // ALLOW_ICMP-NEXT: %[[R_SHIFT:.+]] = comb.mux %[[RHS_0]], %[[R_SHIFT_BY_1]], %lhs
  // ALLOW_ICMP-NEXT: %[[C3_I2:.+]] = hw.constant -2
  // ALLOW_ICMP-NEXT: %[[ICMP:.+]] = comb.icmp ult %rhs, %[[C3_I2]]
  // ALLOW_ICMP-NEXT: %[[R_SHIFT_WITH_BOUND_CHECK:.+]] = comb.mux %[[ICMP]], %[[R_SHIFT]], %[[C0_I2]]
  %1 = comb.shru %lhs, %rhs : i2

  // ALLOW_ICMP-NEXT: %[[LHS_1:.+]] = comb.extract %lhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[LHS_0:.+]] = comb.extract %lhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[CONCAT:.+]] = comb.concat %[[LHS_1]], %[[LHS_0]]
  // ALLOW_ICMP-NEXT: %[[SIGN_REPLICATE:.+]] = comb.replicate %[[LHS_1]] : (i1) -> i2
  // ALLOW_ICMP-NEXT: %[[C1_I2:.+]] = hw.constant 1
  // ALLOW_ICMP-NEXT: %[[ICMP:.+]] = comb.icmp ult %rhs, %[[C1_I2]]
  // ALLOW_ICMP-NEXT: %[[R_SIGNED_SHIFT:.*]] = comb.mux %[[ICMP]], %[[CONCAT]], %[[SIGN_REPLICATE]]
  %2 = comb.shrs %lhs, %rhs : i2

  // ALLOW_ICMP-NEXT: hw.output %[[L_SHIFT_WITH_BOUND_CHECK]], %[[R_SHIFT_WITH_BOUND_CHECK]], %[[R_SIGNED_SHIFT]]
  hw.output %0, %1, %2 : i2, i2, i2
}

// CHECK-LABEL: @divmod
// ALLOW_ICMP-LABEL: @divmod
hw.module @divmod(in %in: i1, in %rhs: i2, out out_divu: i2, out out_modu: i2, out out_divs: i2, out out_mods: i2) {
  %false = hw.constant false
  %lhs = comb.concat %false, %in : i1, i1
  // =DIVU===================================
  // | in | rhs[1] | rhs[0] | divu(=in/rhs) |
  // | ------------------------------------ |
  // | 0  |   0    |   0    |  undef(=0)    |
  // | 0  |   0    |   1    |  0            |
  // | 0  |   1    |   0    |  0            |
  // | 0  |   1    |   1    |  0            |
  // | 1  |   0    |   0    |  undef(=0)    |
  // | 1  |   0    |   1    |  1            |
  // | 1  |   1    |   0    |  0            |
  // | 1  |   1    |   1    |  0            |
  // ========================================
  // ALLOW_ICMP:      %[[C0_I2:.+]] = hw.constant 0 : i2
  // ALLOW_ICMP-NEXT: %[[C1_I2:.+]] = hw.constant 1 : i2
  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[MUX1:.+]] = comb.mux %[[RHS_0]], %[[C1_I2]], %[[C0_I2]] : i2
  // ALLOW_ICMP-NEXT: %[[MUX2:.+]] = comb.mux %[[RHS_1]], %[[C0_I2]], %[[MUX1]] : i2
  // ALLOW_ICMP-NEXT: %[[DIVU:.+]] = comb.mux %in, %[[MUX2]], %[[C0_I2]] : i2
  %0 = comb.divu %lhs, %rhs : i2

  // =MODU===================================
  // | in | rhs[1] | rhs[0] | modu(=in%rhs) |
  // | ------------------------------------ |
  // | 0  |   0    |   0    |  undef(=0)    |
  // | 0  |   0    |   1    |  0            |
  // | 0  |   1    |   0    |  0            |
  // | 0  |   1    |   1    |  0            |
  // | 1  |   0    |   0    |  undef(=0)    |
  // | 1  |   0    |   1    |  0            |
  // | 1  |   1    |   0    |  1            |
  // | 1  |   1    |   1    |  1            |
  // ========================================
  // ALLOW_ICMP: %[[C0_I2:.+]] = hw.constant 0 : i2
  // ALLOW_ICMP-NEXT: %[[C1_I2:.+]] = hw.constant 1 : i2
  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[MUX1:.+]] = comb.mux %[[RHS_1]], %[[C1_I2]], %[[C0_I2]] : i2
  // ALLOW_ICMP-NEXT: %[[MODU:.+]] = comb.mux %in, %[[MUX1]], %[[C0_I2]] : i2
  %1 = comb.modu %lhs, %rhs : i2

  // =DIVS===================================
  // | in | rhs[1] | rhs[0] | divs(=in/rhs) |
  // | ------------------------------------ |
  // | 0  |   0    |   0    |  undef(=0)    |
  // | 0  |   0    |   1    |  0            |
  // | 0  |   1    |   0    |  0            |
  // | 0  |   1    |   1    |  0            |
  // | 1  |   0    |   0    |  undef(=0)    |
  // | 1  |   0    |   1    |  1            |
  // | 1  |   1    |   0    |  0            |
  // | 1  |   1    |   1    |  -1           |
  // ========================================
  // ALLOW_ICMP:      %[[C0_I2:.+]] = hw.constant 0 : i2
  // ALLOW_ICMP-NEXT: %[[C1_I2:.+]] = hw.constant 1 : i2
  // ALLOW_ICMP-NEXT: %[[C_MINUS_1_I2:.+]] = hw.constant -1 : i2
  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[MUX1:.+]] = comb.mux %[[RHS_0]], %[[C_MINUS_1_I2]], %[[C0_I2]] : i2
  // ALLOW_ICMP-NEXT: %[[MUX2:.+]] = comb.mux %[[RHS_0]], %[[C1_I2]], %[[C0_I2]] : i2
  // ALLOW_ICMP-NEXT: %[[MUX3:.+]] = comb.mux %[[RHS_1]], %[[MUX1]], %[[MUX2]] : i2
  // ALLOW_ICMP-NEXT: %[[DIVS:.+]] = comb.mux %in, %[[MUX3]], %[[C0_I2]] : i2
  %2 = comb.divs %lhs, %rhs : i2

  // =MODS===================================
  // | in | rhs[1] | rhs[0] | mods(=in%rhs) |
  // | ------------------------------------ |
  // | 0  |   0    |   0    |  undef(=0)    |
  // | 0  |   0    |   1    |  0            |
  // | 0  |   1    |   0    |  0            |
  // | 0  |   1    |   1    |  0            |
  // | 1  |   0    |   0    |  undef(=0)    |
  // | 1  |   0    |   1    |  0            |
  // | 1  |   1    |   0    |  1            |
  // | 1  |   1    |   1    |  0            |
  // ========================================
  // ALLOW_ICMP:      %[[C0_I2:.+]] = hw.constant 0 : i2
  // ALLOW_ICMP-NEXT: %[[C1_I2:.+]] = hw.constant 1 : i2
  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[MUX1:.+]] = comb.mux %[[RHS_0]], %[[C0_I2]], %[[C1_I2]] : i2
  // ALLOW_ICMP-NEXT: %[[MUX2:.+]] = comb.mux %[[RHS_1]], %[[MUX1]], %[[C0_I2]] : i2
  // ALLOW_ICMP-NEXT: %[[MODS:.+]] = comb.mux %in, %[[MUX2]], %[[C0_I2]] : i2
  %3 = comb.mods %lhs, %rhs : i2

  // ALLOW_ICMP-NEXT: hw.output %[[DIVU]], %[[MODU]], %[[DIVS]], %[[MODS]]
  hw.output %0, %1, %2, %3 : i2, i2, i2, i2
}


// CHECK-LABEL: @divmodu_power_of_two
// ALLOW_ICMP-LABEL: @divmodu_power_of_two
hw.module @divmodu_power_of_two(in %lhs: i8, out out_divu: i8, out out_modu: i8) {
  %rhs = hw.constant 8 : i8
  %0 = comb.divu %lhs, %rhs : i8
  %1 = comb.modu %lhs, %rhs : i8
  // ALLOW_ICMP:      %[[UPPER_5:.+]] = comb.extract %lhs from 3 : (i8) -> i5
  // ALLOW_ICMP-NEXT: %[[C0_I3:.+]] = hw.constant 0 : i3
  // ALLOW_ICMP-NEXT: %[[DIVU:.+]] = comb.concat %[[C0_I3]], %[[UPPER_5]] : i3, i5
  // ALLOW_ICMP-NEXT: %[[LOWER_3:.+]] = comb.extract %lhs from 0 : (i8) -> i3
  // ALLOW_ICMP-NEXT: %[[C0_I5:.+]] = hw.constant 0 : i5
  // ALLOW_ICMP-NEXT: %[[MODU:.+]] = comb.concat %[[C0_I5]], %[[LOWER_3]] : i5, i3
  // ALLOW_ICMP-NEXT: hw.output %[[DIVU]], %[[MODU]] : i8, i8
  hw.output %0, %1 : i8, i8
}

