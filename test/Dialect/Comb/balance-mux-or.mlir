// RUN: circt-opt %s -comb-balance-mux | FileCheck %s

// CHECK-LABEL: hw.module @OrOfMuxes
// CHECK: %[[C0:.*]] = comb.icmp eq %c, %c0_i4
// CHECK: %[[C1:.*]] = comb.icmp eq %c, %c1_i4
// CHECK: %[[X:.*]] = comb.mux bin %[[C1]], %a1, %c0_i4
// CHECK: %[[Y:.*]] = comb.mux bin %[[C0]], %a0, %[[X]]
// CHECK: hw.output %[[Y]]
hw.module @OrOfMuxes(in %c: i4, in %a0: i4, in %a1: i4, out y: i4) {
  %cst0_i4 = hw.constant 0 : i4
  %cst1_i4 = hw.constant 1 : i4
  %c0 = comb.icmp eq %c, %cst0_i4 : i4
  %c1 = comb.icmp eq %c, %cst1_i4 : i4
  %m0 = comb.mux bin %c0, %a0, %cst0_i4 : i4
  %m1 = comb.mux bin %c1, %a1, %cst0_i4 : i4
  %or = comb.or bin %m0, %m1 : i4
  hw.output %or : i4
}

// CHECK-LABEL: hw.module @OrOfMuxesNot2State
// CHECK: comb.or
hw.module @OrOfMuxesNot2State(in %c0: i1, in %c1: i1, in %a0: i4, in %a1: i4, out y: i4) {
  %c0_i4 = hw.constant 0 : i4
  %m0 = comb.mux %c0, %a0, %c0_i4 : i4
  %m1 = comb.mux %c1, %a1, %c0_i4 : i4
  %or = comb.or %m0, %m1 : i4
  hw.output %or : i4
}

// CHECK-LABEL: hw.module @OrOfMuxesNotGuaranteedIndependent
// CHECK: comb.or
hw.module @OrOfMuxesNotGuaranteedIndependent(in %c0: i1, in %c1: i1, in %a0: i4, in %a1: i4, out y: i4) {
  %c0_i4 = hw.constant 0 : i4
  %m0 = comb.mux bin %c0, %a0, %c0_i4 : i4
  %m1 = comb.mux bin %c1, %a1, %c0_i4 : i4
  %or = comb.or %m0, %m1 : i4
  hw.output %or : i4
}
