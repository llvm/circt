// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(comb-balance-mux{mux-chain-threshold=4}))' | FileCheck %s
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(comb-balance-mux{mux-chain-threshold=2}))' | FileCheck %s --check-prefix=MIN2

// CHECK-LABEL: hw.module @priori_mux_false_side_chain
hw.module @priori_mux_false_side_chain(in %cond0: i1, in %cond1: i1, in %cond2: i1, in %cond3: i1, in %cond4: i1, out output: i32) {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c3_i32 = hw.constant 3 : i32
  %c4_i32 = hw.constant 4 : i32
  %c5_i32 = hw.constant 5 : i32
  // Create a long chain of muxes (false-side chain)
  // mux(c0, r0, mux(c1, r1, mux(c2, r2, ...)))
  // => mux(or(c0, c1, c2), mux(or(c0, c1), mux(c0, r0, r1), r2), mux(..))
  %mux5 = comb.mux %cond4, %c5_i32, %c0_i32 : i32
  %mux4 = comb.mux %cond3, %c4_i32, %mux5 : i32
  %mux3 = comb.mux %cond2, %c3_i32, %mux4 : i32
  %mux2 = comb.mux %cond1, %c2_i32, %mux3 : i32
  %mux1 = comb.mux %cond0, %c1_i32, %mux2 : i32
  // CHECK:      %[[MUX0:.*]] = comb.mux %cond0, %c1_i32, %c2_i32 : i32
  // CHECK-NEXT: %[[OR1:.*]] = comb.or bin %cond0, %cond1 : i1
  // CHECK-NEXT: %[[MUX1:.*]] = comb.mux %[[OR1]], %[[MUX0]], %c3_i32 : i32
  // CHECK-NEXT: %[[MUX2:.*]] = comb.mux %cond4, %c5_i32, %c0_i32 : i32
  // CHECK-NEXT: %[[MUX3:.*]] = comb.mux %cond3, %c4_i32, %[[MUX2]] : i32
  // CHECK-NEXT: %[[OR2:.*]] = comb.or bin %cond0, %cond1, %cond2 : i1
  // CHECK-NEXT: %[[MUX4:.*]] = comb.mux %[[OR2]], %[[MUX1]], %[[MUX3]] : i32
  // CHECK-NEXT: hw.output %[[MUX4]] : i32
  hw.output %mux1 : i32
}

// CHECK-LABEL: hw.module @priori_mux_true_side_chain
hw.module @priori_mux_true_side_chain(in %cond0: i1, in %cond1: i1, in %cond2: i1, in %cond3: i1, in %cond4: i1, in %cond5: i1, out output: i32) {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c3_i32 = hw.constant 3 : i32
  %c4_i32 = hw.constant 4 : i32
  %c5_i32 = hw.constant 5 : i32

  // Create a long chain of muxes (true-side chain)
  // mux(c0, mux(c1, mux(c2, ...), r1), r0)
  // => mux(or(!c0, !c1, !c2), mux(or(!c0, !c1), mux(!c0, r0, r1), r2), mux(...))
  // CHECK:      %[[XOR0:.*]] = comb.xor bin %cond0, %true : i1
  // CHECK-NEXT: %[[XOR1:.*]] = comb.xor bin %cond1, %true : i1
  // CHECK-NEXT: %[[XOR2:.*]] = comb.xor bin %cond2, %true : i1
  // CHECK-NEXT: %[[XOR3:.*]] = comb.xor bin %cond3, %true : i1
  // CHECK-NEXT: %[[XOR4:.*]] = comb.xor bin %cond4, %true : i1
  // CHECK-NEXT: %[[MUX0:.*]] = comb.mux %[[XOR0]], %c1_i32, %c2_i32 : i32
  // CHECK-NEXT: %[[OR0:.*]] = comb.or bin %[[XOR0]], %[[XOR1]] : i1
  // CHECK-NEXT: %[[MUX1:.*]] = comb.mux %[[OR0]], %[[MUX0]], %c3_i32 : i32
  // CHECK-NEXT: %[[MUX2:.*]] = comb.mux %[[XOR4]], %c0_i32, %c5_i32 : i32
  // CHECK-NEXT: %[[MUX3:.*]] = comb.mux %[[XOR3]], %c4_i32, %[[MUX2]] : i32
  // CHECK-NEXT: %[[OR1:.*]] = comb.or bin %[[XOR0]], %[[XOR1]], %[[XOR2]] : i1
  %mux5 = comb.mux %cond4, %c5_i32, %c0_i32 : i32
  %mux4 = comb.mux %cond3, %mux5, %c4_i32 : i32
  %mux3 = comb.mux %cond2, %mux4, %c3_i32 : i32
  %mux2 = comb.mux %cond1, %mux3, %c2_i32 : i32
  %mux1 = comb.mux %cond0, %mux2, %c1_i32 : i32

  hw.output %mux1 : i32
}


// CHECK-LABEL: hw.module @test_duplicate
hw.module @test_duplicate(in %cond0: i1, in %cond1: i1, in %cond2: i1, in %cond3: i1, in %cond4: i1, in %cond5: i1, in %cond6: i1, out output: i32) {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c3_i32 = hw.constant 3 : i32
  %c4_i32 = hw.constant 4 : i32

  // mux2_dup's condition is duplicated in the tree.
  // Ensure that the duplicated condition is only evaluated once.
  // CHECK:      %[[MUX0:.*]] = comb.mux %cond0, %c1_i32, %c2_i32 : i32
  // CHECK-NEXT: %[[MUX1:.*]] = comb.mux %cond3, %c4_i32, %c0_i32 : i32
  // CHECK-NEXT: %[[MUX2:.*]] = comb.mux %cond2, %c3_i32, %[[MUX1]] : i32
  // CHECK-NEXT: %[[OR:.*]] = comb.or bin %cond0, %cond1 : i1
  // CHECK-NEXT: %[[MUX3:.*]] = comb.mux %[[OR]], %[[MUX0]], %[[MUX2]] : i32
  // CHECK-NEXT: hw.output %[[MUX3]] : i32
  %mux4 = comb.mux %cond3, %c4_i32, %c0_i32 : i32
  %mux2_dup = comb.mux %cond0, %c3_i32, %mux4 : i32
  %mux3 = comb.mux %cond2, %c3_i32, %mux2_dup : i32
  %mux2 = comb.mux %cond1, %c2_i32, %mux3 : i32
  %mux1 = comb.mux %cond0, %c1_i32, %mux2 : i32
  hw.output %mux1 : i32
}

// CHECK-LABEL: hw.module @test_index_comparison
hw.module @test_index_comparison(in %index: i4, out output: i32) {
  %c0_i4 = hw.constant 0 : i4
  %c1_i4 = hw.constant 1 : i4
  %c2_i4 = hw.constant 2 : i4
  %c3_i4 = hw.constant 3 : i4
  %c4_i4 = hw.constant 4 : i4
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c3_i32 = hw.constant 3 : i32
  %c4_i32 = hw.constant 4 : i32

  // Create a mux chain that compares index against constants
  // This represents:
  //   if (index == 0) 0
  //   else if (index == 1) 1
  //   else if (index == 2) 2
  //   else if (index == 3) 3
  //   else 4
  %cmp0 = comb.icmp eq %index, %c0_i4 : i4
  %cmp1 = comb.icmp eq %index, %c1_i4 : i4
  %cmp2 = comb.icmp eq %index, %c2_i4 : i4
  %cmp3 = comb.icmp eq %index, %c3_i4 : i4
  %mux4 = comb.mux %cmp3, %c3_i32, %c4_i32 : i32
  %mux3 = comb.mux %cmp2, %c2_i32, %mux4 : i32
  %mux2 = comb.mux %cmp1, %c1_i32, %mux3 : i32
  %mux1 = comb.mux %cmp0, %c0_i32, %mux2 : i32
  // CHECK:     %[[EXTRACT0:.*]] = comb.extract %index from 0 : (i4) -> i1
  // CHECK-NET: %[[EXTRACT1:.*]] = comb.extract %index from 1 : (i4) -> i1
  // CHECK-NET: %[[EXTRACT2:.*]] = comb.extract %index from 2 : (i4) -> i1
  // CHECK-NET: %[[EXTRACT3:.*]] = comb.extract %index from 3 : (i4) -> i1
  // CHECK-NET: %[[MUX0:.*]] = comb.mux %[[EXTRACT0]], %c3_i32, %c2_i32 : i32
  // CHECK-NET: %[[MUX1:.*]] = comb.mux %[[EXTRACT0]], %c1_i32, %c0_i32 : i32
  // CHECK-NET: %[[MUX2:.*]] = comb.mux %[[EXTRACT1]], %[[MUX0]], %[[MUX1]] : i32
  // CHECK-NET: %[[OR:.*]] = comb.or bin %[[EXTRACT3]], %[[EXTRACT2]] : i1
  // CHECK-NET: %[[MUX3:.*]] = comb.mux %[[OR]], %c4_i32, %[[MUX2]] : i32

  hw.output %mux1 : i32
}

// CHECK-LABEL: hw.module @test_min_conditions_2
// CHECK-NOT: comb.or
// MIN2-LABEL: hw.module @test_min_conditions_2
// MIN2: [[OR:%.*]] = comb.or
// MIN2: comb.mux [[OR]]
hw.module @test_min_conditions_2(in %cond0: i1, in %cond1: i1, in %cond2: i1, out output: i32) {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c3_i32 = hw.constant 3 : i32

  // Chain with 3 conditions - should be rebalanced with --balance-mux='min-conditions=2'
  %mux3 = comb.mux %cond2, %c3_i32, %c0_i32 : i32
  %mux2 = comb.mux %cond1, %c2_i32, %mux3 : i32
  %mux1 = comb.mux %cond0, %c1_i32, %mux2 : i32

  hw.output %mux1 : i32
}
