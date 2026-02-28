// RUN: circt-opt %s --comb-simplify-tt | FileCheck %s

// CHECK-LABEL: @truth_table_constant_true
hw.module @truth_table_constant_true(in %a: i1, in %b: i1, out out: i1) {
  // Truth table that is always true (all ones)
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: hw.output [[TRUE]]
  %0 = comb.truth_table %a, %b -> [true, true, true, true]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_constant_false
hw.module @truth_table_constant_false(in %a: i1, in %b: i1, out out: i1) {
  // Truth table that is always false (all zeros)
  // CHECK-NEXT: [[FALSE:%.+]] = hw.constant false
  // CHECK-NEXT: hw.output [[FALSE]]
  %0 = comb.truth_table %a, %b -> [false, false, false, false]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_identity
hw.module @truth_table_identity(in %a: i1, in %b: i1, in %c: i1, out out: i1) {
  // Truth table that depends only on %a
  // Pattern: [0,0,0,0,1,1,1,1] means output follows first input
  // CHECK-NEXT: hw.output %a
  %0 = comb.truth_table %a, %b, %c -> [false, false, false, false, true, true, true, true]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_inverted
hw.module @truth_table_inverted(in %a: i1, in %b: i1, in %c: i1, out out: i1) {
  // Truth table that depends only on %a (inverted)
  // Pattern: [1,1,1,1,0,0,0,0] means output is NOT of first input
  // Should simplify to single-input truth table for negation
  // CHECK-NEXT: [[NOT:%.+]] = comb.truth_table %a -> [true, false]
  // CHECK-NEXT: hw.output [[NOT]]
  %0 = comb.truth_table %a, %b, %c -> [true, true, true, true, false, false, false, false]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_middle_input_identity
hw.module @truth_table_middle_input_identity(in %a: i1, in %b: i1, in %c: i1, out out: i1) {
  // Truth table that depends only on %b (middle input, identity)
  // Pattern: [0,0,1,1,0,0,1,1] means output follows second input
  // CHECK-NEXT: hw.output %b
  %0 = comb.truth_table %a, %b, %c -> [false, false, true, true, false, false, true, true]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_middle_input_inverted
hw.module @truth_table_middle_input_inverted(in %a: i1, in %b: i1, in %c: i1, out out: i1) {
  // Truth table that depends only on %b (middle input, inverted)
  // Pattern: [1,1,0,0,1,1,0,0] means output is NOT of second input
  // Should simplify to single-input truth table for negation
  // CHECK-NEXT: [[NOT:%.+]] = comb.truth_table %b -> [true, false]
  // CHECK-NEXT: hw.output [[NOT]]
  %0 = comb.truth_table %a, %b, %c -> [true, true, false, false, true, true, false, false]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_last_input_identity
hw.module @truth_table_last_input_identity(in %a: i1, in %b: i1, in %c: i1, out out: i1) {
  // Truth table that depends only on %c (last input, identity)
  // Pattern: [0,1,0,1,0,1,0,1] means output follows third input
  // CHECK-NEXT: hw.output %c
  %0 = comb.truth_table %a, %b, %c -> [false, true, false, true, false, true, false, true]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_last_input_inverted
hw.module @truth_table_last_input_inverted(in %a: i1, in %b: i1, in %c: i1, out out: i1) {
  // Truth table that depends only on %c (last input, inverted)
  // Pattern: [1,0,1,0,1,0,1,0] means output is NOT of third input
  // Should simplify to single-input truth table for negation
  // CHECK-NEXT: [[NOT:%.+]] = comb.truth_table %c -> [true, false]
  // CHECK-NEXT: hw.output [[NOT]]
  %0 = comb.truth_table %a, %b, %c -> [true, false, true, false, true, false, true, false]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_two_input_non_foldable
hw.module @truth_table_two_input_non_foldable(in %a: i1, in %b: i1, out out: i1) {
  // Truth table depends on both inputs, Should not be canonicalized
  // CHECK-NEXT: %0 = comb.truth_table %a, %b -> [false, false, false, true]
  // CHECK-NEXT: hw.output %0
  %0 = comb.truth_table %a, %b -> [false, false, false, true]
  hw.output %0 : i1
}

// CHECK-LABEL: @truth_table_with_extract_operations
hw.module @truth_table_with_extract_operations(in %c: i3, out out: i1) {
  // Truth table depends only on first input (%2 = LSB of %c)
  // CHECK: [[TMP:%.+]] = comb.extract %c from 0
  // CHECK: hw.output [[TMP]]
  %0 = comb.extract %c from 2 : (i3) -> i1
  %1 = comb.extract %c from 1 : (i3) -> i1
  %2 = comb.extract %c from 0 : (i3) -> i1
  %3 = comb.truth_table %2, %0, %1 -> [false, false, false, false, true, true, true, true]
  hw.output %3 : i1
}
