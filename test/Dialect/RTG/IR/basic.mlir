// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: rtg.sequence @seq
// CHECK-SAME: attributes {rtg.some_attr} {
rtg.sequence @seq0 attributes {rtg.some_attr} {
}

// CHECK-LABEL: rtg.sequence @seq1
// CHECK: ^bb0(%arg0: i32, %arg1: !rtg.sequence):
rtg.sequence @seq1 {
^bb0(%arg0: i32, %arg1: !rtg.sequence):
}

// CHECK-LABEL: rtg.sequence @invocations
rtg.sequence @invocations {
  // CHECK: [[V0:%.+]] = rtg.sequence_closure @seq0
  // CHECK: [[C0:%.+]] = arith.constant 0 : i32
  // CHECK: [[V1:%.+]] = rtg.sequence_closure @seq1([[C0]], [[V0]] : i32, !rtg.sequence)
  // CHECK: rtg.invoke_sequence [[V0]]
  // CHECK: rtg.invoke_sequence [[V1]]
  %0 = rtg.sequence_closure @seq0
  %c0_i32 = arith.constant 0 : i32
  %1 = rtg.sequence_closure @seq1(%c0_i32, %0 : i32, !rtg.sequence)
  rtg.invoke_sequence %0
  rtg.invoke_sequence %1
}
