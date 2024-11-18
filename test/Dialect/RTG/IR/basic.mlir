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
