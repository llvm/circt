// RUN: circt-opt %s --datapath-reduce-delay | FileCheck %s

// CHECK-LABEL: @do_nothing
hw.module @do_nothing(in %a : i4, in %b : i4, in %c : i4, in %sel : i1, out res : i4) {
  // CHECK-NEXT: %[[MUX:.+]] = comb.mux %sel, %a, %b : i4
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %[[MUX]], %c : i4
  // CHECK-NEXT: hw.output %[[ADD]] : i4
  %0 = comb.mux %sel, %a, %b : i4
  %1 = comb.add %0, %c : i4
  hw.output %1 : i4
}

// CHECK-LABEL: @fold_adds
hw.module @fold_adds(in %a : i4, in %b : i4, in %c : i4, in %d : i4, out abc : i4, out abd : i4) {
  // CHECK-NEXT: %[[COMP_ABC:.+]]:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
  // CHECK-NEXT: %[[ABC:.+]] = comb.add bin %[[COMP_ABC]]#0, %[[COMP_ABC]]#1 : i4
  // CHECK-NEXT: %[[COMP_ABD:.+]]:2 = datapath.compress %a, %b, %d : i4 [3 -> 2]
  // CHECK-NEXT: %[[ABD:.+]] = comb.add bin %[[COMP_ABD]]#0, %[[COMP_ABD]]#1 : i4
  // CHECK-NEXT: hw.output %[[ABC]], %[[ABD]] : i4, i4
  %0 = comb.add %a, %b : i4
  %1 = comb.add %0, %c : i4
  %2 = comb.add %0, %d : i4
  hw.output %1, %2 : i4, i4
}

// CHECK-LABEL: @add_mux_left
hw.module @add_mux_left(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %sel : i1, out res : i4) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %[[LHS:.+]] = comb.mux %sel, %a, %c : i4
  // CHECK-NEXT: %[[RHS:.+]] = comb.mux %sel, %b, %c0_i4 : i4
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %[[LHS]], %[[RHS]], %d : i4 [3 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4
  %0 = comb.add %a, %b : i4
  %1 = comb.mux %sel, %0, %c : i4
  %2 = comb.add %1, %d : i4
  hw.output %2 : i4
}

// CHECK-LABEL: @add_mux_right
hw.module @add_mux_right(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %sel : i1, out res : i4) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %[[LHS:.+]] = comb.mux %sel, %a, %b : i4
  // CHECK-NEXT: %[[RHS:.+]] = comb.mux %sel, %c0_i4, %c : i4
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %[[LHS]], %[[RHS]], %d : i4 [3 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4
  %0 = comb.add %b, %c : i4
  %1 = comb.mux %sel, %a, %0 : i4
  %2 = comb.add %1, %d : i4
  hw.output %2 : i4
}

// CHECK-LABEL: @add_mux_both
hw.module @add_mux_both(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %e : i4, in %sel : i1, out res : i4) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %[[ARG0:.+]] = comb.mux %sel, %a, %c : i4
  // CHECK-NEXT: %[[ARG1:.+]] = comb.mux %sel, %b, %d : i4
  // CHECK-NEXT: %[[ARG2:.+]] = comb.mux %sel, %c0_i4, %e : i4
  // CHECK-NEXT: %[[COMP:.+]]:2 = datapath.compress %[[ARG0]], %[[ARG1]], %[[ARG2]], %e : i4 [4 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4
  %0 = comb.add %a, %b : i4
  %1 = comb.add %c, %d, %e : i4
  %2 = comb.mux %sel, %0, %1 : i4
  %3 = comb.add %2, %e : i4
  hw.output %3 : i4
}

// CHECK-LABEL: @add_compare
hw.module @add_compare(in %a : i16, in %b : i16, in %c : i16, out ugt : i1, out uge : i1) {
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %c1_i18 = hw.constant 1 : i18
  // CHECK-NEXT: %c2_i18 = hw.constant 2 : i18
  // CHECK-NEXT: %c-1_i18 = hw.constant -1 : i18
  // CHECK-NEXT: %false = hw.constant false
  // Compute ugt
  // CHECK-NEXT: %[[AEXT1:.+]] = comb.concat %false, %a : i1, i16
  // CHECK-NEXT: %[[BEXT1:.+]] = comb.concat %false, %b : i1, i16
  // CHECK-NEXT: %[[CEXT1:.+]] = comb.concat %false, %c : i1, i16
  // CHECK-NEXT: %[[CEXT2:.+]] = comb.concat %false, %[[CEXT1]] : i1, i17
  // CHECK-NEXT: %[[AEXT2:.+]] = comb.concat %false, %[[AEXT1]] : i1, i17
  // CHECK-NEXT: %[[AINV:.+]] = comb.xor %[[AEXT2]], %c-1_i18 : i18
  // CHECK-NEXT: %[[BEXT2:.+]] = comb.concat %false, %[[BEXT1]] : i1, i17
  // CHECK-NEXT: %[[BINV:.+]] = comb.xor %[[BEXT2]], %c-1_i18 : i18
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %[[CEXT2]], %[[AINV]], %[[BINV]], %c2_i18 : i18
  // CHECK-NEXT: %[[UGT:.+]] = comb.extract %[[ADD]] from 17 : (i18) -> i1
  // Now compute uge
  // CHECK-NEXT: %[[AEXT2:.+]] = comb.concat %false, %0 : i1, i17
  // CHECK-NEXT: %[[BEXT2:.+]] = comb.concat %false, %1 : i1, i17
  // CHECK-NEXT: %[[CEXT2:.+]] = comb.concat %false, %[[CEXT1]] : i1, i17
  // CHECK-NEXT: %[[CINV:.+]] = comb.xor %[[CEXT2]], %c-1_i18 : i18
  // CHECK-NEXT: %[[ADD_UGE:.+]] = comb.add %[[AEXT2]], %[[BEXT2]], %[[CINV]], %c1_i18 : i18
  // CHECK-NEXT: %[[INV_UGE:.+]] = comb.extract %[[ADD_UGE]] from 17 : (i18) -> i1
  // CHECK-NEXT: %[[UGE:.+]] = comb.xor %[[INV_UGE]], %true : i1
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
