// RUN: circt-opt --rtg-elaborate=debug=true --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: rtg.sequence @seq0
rtg.sequence @seq0 {
  %2 = arith.constant 2 : i32
}

// CHECK-LABEL: rtg.sequence @seq2
rtg.sequence @seq2 {
^bb0(%arg0: !rtg.sequence):
  %0 = rtg.sequence_closure @seq0
  %set = rtg.set_create %arg0, %0 : !rtg.sequence
  // expected-warning @below {{set contained 1 duplicate value(s), the value at index 0 might not be the intended one}}
  %seq = rtg.set_select_random %set : !rtg.set<!rtg.sequence> {rtg.elaboration = 0}
  rtg.invoke_sequence %seq
  rtg.invoke_sequence %seq
}

// Test the set operations and passing a sequence ot another one via argument
// CHECK-LABEL: rtg.test @setOperations
rtg.test @setOperations : !rtg.dict<> {
  // CHECK-NEXT: arith.constant 2 : i32
  // CHECK-NEXT: arith.constant 2 : i32
  // CHECK-NEXT: }
  %0 = rtg.sequence_closure @seq0
  %1 = rtg.sequence_closure @seq2(%0 : !rtg.sequence)
  %set = rtg.set_create %0, %1 : !rtg.sequence
  %seq = rtg.set_select_random %set : !rtg.set<!rtg.sequence> {rtg.elaboration = 0}
  %new_set = rtg.set_create %seq : !rtg.sequence
  %diff = rtg.set_difference %set, %new_set : !rtg.set<!rtg.sequence>
  %seq1 = rtg.set_select_random %diff : !rtg.set<!rtg.sequence> {rtg.elaboration = 0}
  rtg.invoke_sequence %seq1
}

// CHECK-LABEL: rtg.sequence @seq3
rtg.sequence @seq3 {
^bb0(%arg0: !rtg.set<!rtg.sequence>):
  %seq = rtg.set_select_random %arg0 : !rtg.set<!rtg.sequence> {rtg.elaboration = 0}
  rtg.invoke_sequence %seq
}

// CHECK-LABEL: rtg.test @setArguments
rtg.test @setArguments : !rtg.dict<> {
  // CHECK-NEXT: arith.constant 2 : i32
  // CHECK-NEXT: arith.constant 2 : i32
  // CHECK-NEXT: }
  %0 = rtg.sequence_closure @seq0
  %1 = rtg.sequence_closure @seq2(%0 : !rtg.sequence)
  %2 = rtg.set_create %1, %0 : !rtg.sequence
  %3 = rtg.sequence_closure @seq3(%2 : !rtg.set<!rtg.sequence>)
  rtg.invoke_sequence %3
}

// CHECK-LABEL: rtg.sequence @seq4
rtg.sequence @seq4 {
^bb0(%arg0: !rtg.sequence):
  %0 = rtg.sequence_closure @seq0
  %set = rtg.set_create %arg0, %0 : !rtg.sequence
}

// Make sure we also delete ops that don't have any users and thus could be
// skipped and end up with null operands because the defining op was deleted due
// to other users.
// CHECK-LABEL: rtg.test @noNullOperands
rtg.test @noNullOperands : !rtg.dict<> {
  // CHECK-NEXT: }
  %1 = rtg.sequence_closure @seq0
  %2 = rtg.sequence_closure @seq4(%1 : !rtg.sequence)
  rtg.invoke_sequence %2
}

rtg.target @target0 : !rtg.dict<num_cpus: i32> {
  %0 = arith.constant 0 : i32
  rtg.yield %0 : i32
}

rtg.target @target1 : !rtg.dict<num_cpus: i32> {
  %0 = arith.constant 1 : i32
  rtg.yield %0 : i32
}

// CHECK-LABEL: @targetTest_target0
// CHECK: [[V0:%.+]] = arith.constant 0
// CHECK: arith.addi [[V0]], [[V0]]

// CHECK-LABEL: @targetTest_target1
// CHECK: [[V0:%.+]] = arith.constant 1
// CHECK: arith.addi [[V0]], [[V0]]
rtg.test @targetTest : !rtg.dict<num_cpus: i32> {
^bb0(%arg0: i32):
  arith.addi %arg0, %arg0 : i32
}

// CHECK-NOT: @unmatchedTest
rtg.test @unmatchedTest : !rtg.dict<num_cpus: i64> {
^bb0(%arg0: i64):
  arith.addi %arg0, %arg0 : i64
}

// -----

rtg.test @opaqueValuesAndSets : !rtg.dict<> {
  %0 = arith.constant 2 : i32
  // expected-error @below {{cannot create a set of opaque values because they cannot be reliably uniqued}}
  %1 = rtg.set_create %0 : i32
}

// -----

rtg.sequence @seq0 {
  %2 = arith.constant 2 : i32
}

// Test that the elaborator value interning works as intended and exercise 'set_select_random' error messages.
rtg.test @setOperations : !rtg.dict<> {
  %0 = rtg.sequence_closure @seq0
  %1 = rtg.sequence_closure @seq0
  %set = rtg.set_create %0, %1 : !rtg.sequence
  // expected-warning @below {{set contained 1 duplicate value(s), the value at index 2 might not be the intended one}}
  // expected-error @below {{'rtg.elaboration' attribute value out of bounds, must be between 0 (incl.) and 2 (excl.)}}
  %seq = rtg.set_select_random %set : !rtg.set<!rtg.sequence> {rtg.elaboration = 2}
  rtg.invoke_sequence %seq
}
