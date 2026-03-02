// RUN: circt-opt %s --ltl-infer-clocks | FileCheck %s

// CHECK-LABEL: hw.module @InferClockFromClockOp
hw.module @InferClockFromClockOp(in %clk: i1, in %a: i1, in %b: i1) {
  %true = hw.constant true
  
  // Before: delay uses placeholder clock (%true)
  // After: delay should use %clk from the enclosing ltl.clock
  
  // CHECK: [[DELAY:%.+]] = ltl.delay %clk, posedge, %a, 1, 0 : i1
  // CHECK-NOT: ltl.delay %true
  %d = ltl.delay %true, posedge, %a, 1, 0 : i1
  
  // CHECK: ltl.clock [[DELAY]], posedge %clk
  %p = ltl.clock %d, posedge %clk : !ltl.sequence
  
  sv.assert_property %p : !ltl.sequence
}

// CHECK-LABEL: hw.module @InferClockNested
hw.module @InferClockNested(in %clk: i1, in %a: i1, in %b: i1) {
  %true = hw.constant true
  
  // Multiple delays in a chain should all get the clock
  // CHECK: [[D1:%.+]] = ltl.delay %clk, posedge, %a, 1, 0 : i1
  %d1 = ltl.delay %true, posedge, %a, 1, 0 : i1
  
  // CHECK: [[D2:%.+]] = ltl.delay %clk, posedge, %b, 2, 0 : i1
  %d2 = ltl.delay %true, posedge, %b, 2, 0 : i1
  
  // CHECK: [[CONCAT:%.+]] = ltl.concat [[D1]], [[D2]]
  %c = ltl.concat %d1, %d2 : !ltl.sequence, !ltl.sequence
  
  // CHECK: ltl.clock [[CONCAT]], posedge %clk
  %p = ltl.clock %c, posedge %clk : !ltl.sequence
  
  sv.assert_property %p : !ltl.sequence
}

// CHECK-LABEL: hw.module @PreserveRealClock
hw.module @PreserveRealClock(in %clk: i1, in %clk2: i1, in %a: i1) {
  // Delay already has a real clock - should NOT be modified
  // CHECK: [[DELAY:%.+]] = ltl.delay %clk2, posedge, %a, 1, 0 : i1
  %d = ltl.delay %clk2, posedge, %a, 1, 0 : i1
  
  // The clock op has a different clock, but we don't change the delay
  // because it already has a non-placeholder clock
  // CHECK: ltl.clock [[DELAY]], posedge %clk
  %p = ltl.clock %d, posedge %clk : !ltl.sequence
  
  sv.assert_property %p : !ltl.sequence
}

// CHECK-LABEL: hw.module @InferEdge
hw.module @InferEdge(in %clk: i1, in %a: i1) {
  %true = hw.constant true
  
  // The delay should get the edge from the clock op (negedge)
  // CHECK: [[DELAY:%.+]] = ltl.delay %clk, negedge, %a, 1, 0 : i1
  %d = ltl.delay %true, posedge, %a, 1, 0 : i1
  
  // CHECK: ltl.clock [[DELAY]], negedge %clk
  %p = ltl.clock %d, negedge %clk : !ltl.sequence
  
  sv.assert_property %p : !ltl.sequence
}

// CHECK-LABEL: hw.module @NoClockOp
hw.module @NoClockOp(in %a: i1) {
  %true = hw.constant true
  
  // No clock op wrapping - delay keeps placeholder clock
  // CHECK: ltl.delay %true, posedge, %a, 1, 0 : i1
  %d = ltl.delay %true, posedge, %a, 1, 0 : i1
  
  sv.assert_property %d : !ltl.sequence
}

// CHECK-LABEL: hw.module @MultiClockSharedDelay
// Test P2: A single delay used by multiple clock ops with different clocks
// Each clock op should get its own copy of the delay with the correct clock
hw.module @MultiClockSharedDelay(in %clk1: i1, in %clk2: i1, in %a: i1) {
  %true = hw.constant true
  
  // Original delay with placeholder clock - used by two different clock ops
  %d = ltl.delay %true, posedge, %a, 1, 0 : i1
  
  // CHECK-DAG: [[D1:%.+]] = ltl.delay %clk1, posedge, %a, 1, 0 : i1
  // CHECK-DAG: [[D2:%.+]] = ltl.delay %clk2, negedge, %a, 1, 0 : i1
  
  // CHECK-DAG: [[P1:%.+]] = ltl.clock [[D1]], posedge %clk1
  %p1 = ltl.clock %d, posedge %clk1 : !ltl.sequence
  
  // CHECK-DAG: [[P2:%.+]] = ltl.clock [[D2]], negedge %clk2
  %p2 = ltl.clock %d, negedge %clk2 : !ltl.sequence
  
  sv.assert_property %p1 : !ltl.sequence
  sv.assert_property %p2 : !ltl.sequence
}

// CHECK-LABEL: hw.module @MultiClockSharedChain
// Test P2: A chain of ops used by multiple clock ops
hw.module @MultiClockSharedChain(in %clk1: i1, in %clk2: i1, in %a: i1, in %b: i1) {
  %true = hw.constant true
  
  // Shared chain: delay -> concat
  %d1 = ltl.delay %true, posedge, %a, 1, 0 : i1
  %d2 = ltl.delay %true, posedge, %b, 2, 0 : i1
  %c = ltl.concat %d1, %d2 : !ltl.sequence, !ltl.sequence
  
  // Both clock ops use the same concat
  // CHECK-DAG: ltl.delay %clk1, posedge, %a, 1, 0 : i1
  // CHECK-DAG: ltl.delay %clk1, posedge, %b, 2, 0 : i1
  // CHECK-DAG: ltl.delay %clk2, posedge, %a, 1, 0 : i1
  // CHECK-DAG: ltl.delay %clk2, posedge, %b, 2, 0 : i1
  
  %p1 = ltl.clock %c, posedge %clk1 : !ltl.sequence
  %p2 = ltl.clock %c, posedge %clk2 : !ltl.sequence
  
  sv.assert_property %p1 : !ltl.sequence
  sv.assert_property %p2 : !ltl.sequence
}
