// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-comb-to-aig{additional-legal-ops=comb.xor,comb.or,comb.and,comb.mux}))" | FileCheck %s
// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-comb-to-aig{additional-legal-ops=comb.xor,comb.or,comb.and,comb.mux,comb.add}))" | FileCheck %s --check-prefix=ALLOW_ADD


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
