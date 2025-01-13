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

// CHECK-LABEL: @sub
// ALLOW_ADD-LABEL: @sub
// ALLOW_ADD-NEXT: %[[NOT_RHS:.+]] = aig.and_inv not %rhs
// ALLOW_ADD-NEXT: %[[CONST:.+]] = hw.constant 1 : i4
// ALLOW_ADD-NEXT: %[[ADD:.+]] = comb.add bin %lhs, %[[NOT_RHS]], %[[CONST]]
// ALLOW_ADD-NEXT: hw.output %[[ADD]]
hw.module @sub(in %lhs: i4, in %rhs: i4, out out: i4) {
  %0 = comb.sub %lhs, %rhs : i4
  hw.output %0 : i4
}


// CHECK-LABEL: @mul
// ALLOW_ADD-LABEL: @mul
// ALLOW_ADD-NEXT:   %[[EXT_0:.+]] = comb.extract %lhs from 0 : (i2) -> i1
// ALLOW_ADD-NEXT:   %[[EXT_1:.+]] = comb.extract %lhs from 1 : (i2) -> i1
// ALLOW_ADD-NEXT:   %c0_i2 = hw.constant 0 : i2
// ALLOW_ADD-NEXT:   %[[MUX_0:.+]] = comb.mux %[[EXT_0]], %rhs, %c0_i2 : i2
// ALLOW_ADD-NEXT:   %[[MUX_1:.+]] = comb.mux %[[EXT_1]], %rhs, %c0_i2 : i2
// ALLOW_ADD-NEXT:   %[[EXT_MUX_1:.+]] = comb.extract %[[MUX_1]] from 0 : (i2) -> i1
// ALLOW_ADD-NEXT:   %false = hw.constant false
// ALLOW_ADD-NEXT:   %[[SHIFT:.+]] = comb.concat %[[EXT_MUX_1]], %false : i1, i1
// ALLOW_ADD-NEXT:   %[[ADD:.+]] = comb.add bin %[[MUX_0]], %[[SHIFT]] : i2
// ALLOW_ADD-NEXT:   hw.output %[[ADD]] : i2
// ALLOW_ADD-NEXT: }
hw.module @mul(in %lhs: i2, in %rhs: i2, out out: i2) {
  %0 = comb.mul %lhs, %rhs : i2
  hw.output %0 : i2
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
  %0 = comb.shl %lhs, %rhs : i2
  %1 = comb.shru %lhs, %rhs : i2
  %2 = comb.shrs %lhs, %rhs : i2
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
  // ALLOW_ICMP-NEXT: %[[LHS_1:.+]] = comb.extract %lhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[LHS_0:.+]] = comb.extract %lhs from 0 : (i2) -> i1
  // ALLOW_ICMP-NEXT: %[[CONCAT:.+]] = comb.concat %[[LHS_1]], %[[LHS_0]]
  // ALLOW_ICMP-NEXT: %[[SIGN_REPLICATE:.+]] = comb.replicate %[[LHS_1]] : (i1) -> i2
  // ALLOW_ICMP-NEXT: %[[C1_I2:.+]] = hw.constant 1
  // ALLOW_ICMP-NEXT: %[[ICMP:.+]] = comb.icmp ult %rhs, %[[C1_I2]]
  // ALLOW_ICMP-NEXT: %[[R_SIGNED_SHIFT:.*]] = comb.mux %[[ICMP]], %[[CONCAT]], %[[SIGN_REPLICATE]]
  // ALLOW_ICMP-NEXT: hw.output %[[L_SHIFT_WITH_BOUND_CHECK]], %[[R_SHIFT_WITH_BOUND_CHECK]], %[[R_SIGNED_SHIFT]]
  hw.output %0, %1, %2 : i2, i2, i2
}
