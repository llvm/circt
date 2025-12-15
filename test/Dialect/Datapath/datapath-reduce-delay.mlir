// RUN: circt-opt %s --datapath-reduce-delay | FileCheck %s


// CHECK-LABEL: @add_compare
hw.module @add_compare(in %a : i16, in %b : i16, in %c : i16, out ugt : i1, out uge : i1) {
  // CHECK-NEXT: %c2_i18 = hw.constant 2 : i18
  // CHECK-NEXT: %c-1_i18 = hw.constant -1 : i18
  // CHECK-NEXT: %c0_i2 = hw.constant 0 : i2
  // CHECK-NEXT: %false = hw.constant false
  // Compute ugt
  // CHECK-NEXT: %[[AEXT1:.+]] = comb.concat %false, %a : i1, i16
  // CHECK-NEXT: %[[BEXT1:.+]] = comb.concat %false, %b : i1, i16
  // CHECK-NEXT: %[[CEXT2:.+]] = comb.concat %c0_i2, %c : i2, i16
  // CHECK-NEXT: %[[AEXT2:.+]] = comb.concat %false, %[[AEXT1]] : i1, i17
  // CHECK-NEXT: %[[AINV:.+]] = comb.xor bin %[[AEXT2]], %c-1_i18 : i18
  // CHECK-NEXT: %[[BEXT2:.+]] = comb.concat %false, %[[BEXT1]] : i1, i17
  // CHECK-NEXT: %[[BINV:.+]] = comb.xor bin %[[BEXT2]], %c-1_i18 : i18
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %[[CEXT2]], %[[AINV]], %[[BINV]], %c2_i18 : i18
  // CHECK-NEXT: %[[UGT:.+]] = comb.extract %[[ADD]] from 17 : (i18) -> i1
  // Now compute uge
  // CHECK-NEXT: %[[AEXT2:.+]] = comb.concat %false, %0 : i1, i17
  // CHECK-NEXT: %[[BEXT2:.+]] = comb.concat %false, %1 : i1, i17
  // CHECK-NEXT: %[[CEXT:.+]] = comb.concat %c0_i2, %c : i2, i16
  // CHECK-NEXT: %[[CINV:.+]] = comb.xor bin %[[CEXT]], %c-1_i18 : i18
  // CHECK-NEXT: %[[ADD_UGE]] = comb.add %[[AEXT2]], %[[BEXT2]], %[[CINV]], %c1_i18 : i18
  // CHECK-NEXT: %[[INV_UGE:.+]] = comb.extract %[[ADD_UGE]] from 17 : (i18) -> i1
  // CHECK-NEXT: %[[UGE:.+]] = comb.xor bin %[[INV_UGE]], %true : i1
  // CHECK-NEXT: hw.output %[[UGT]], %[[UGE]] : i1, i1
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i16
  %1 = comb.concat %false, %b : i1, i16
  %2 = comb.add %0, %1 {comb.nuw} : i17 
  %3 = comb.concat %false, %c : i1, i16
  %4 = comb.icmp ugt %2, %3 : i17
  %5 = comb.icmp uge %2, %3 : i17
  hw.output %4, %5 : i1, i1
}

// CHECK-LABEL: @compare_do_nothing
hw.module @compare_do_nothing(in %a : i16, in %b : i16, in %c : i16, out res1 : i1, out res2 : i1) {
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: %[[AEXT1:.+]] = comb.concat %false, %a : i1, i16
  // CHECK-NEXT: %[[BEXT1:.+]] = comb.concat %false, %b : i1, i16
  // CHECK-NEXT %[[ADD:.+]] = comb.add %0, %1 {comb.nuw} : i17 
  // CHECK-NEXT %[[CEXT:.+]] = comb.concat %false, %c : i1, i16
  // CHECK-NEXT comb.icmp sgt %[[ADD]], %[[CEXT]] : i17
  // CHECK-NEXT comb.icmp ult %a, %b : i16
  %false = hw.constant false
  %0 = comb.concat %false, %a : i1, i16
  %1 = comb.concat %false, %b : i1, i16
  %2 = comb.add %0, %1 {comb.nuw} : i17 
  %3 = comb.concat %false, %c : i1, i16
  %4 = comb.icmp sgt %2, %3 : i17
  %5 = comb.icmp ult %a, %b : i16
  hw.output %4, %5 : i1, i1
}
