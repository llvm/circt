// RUN: circt-opt --rtg-reg-alloc %s | FileCheck %s

// CHECK-LABEL: rtg.sequence @seq0
rtg.sequence @seq0 {
  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  %3 = rtg.label.decl "label_{0}_{1}", %1, %2 : i32, i32 -> i32
  rtg.label %3 : i32
}

// CHECK-LABEL: rtg.sequence @seq1
rtg.sequence @seq1 {
^bb0(%arg0: i32):
  %3 = arith.constant 3 : i32
  rtgtest.instr_a %3, %arg0
}

// CHECK-LABEL: rtg.sequence @seq2
rtg.sequence @seq2 {
  // CHECK-SEED0: [[C50:%.+]] = arith.constant 50 : i32
  %c50 = arith.constant 50 : index
  %c50_i32 = arith.constant 50 : i32
  // CHECK-NOT: rtg.bag_select_random
  // CHECK-SEED0: rtgtest.instr_a %{{.*}}, [[C50]]
  // CHECK-SEED1: rtg.label.decl
  // CHECK: rtg.sequence_closure @seq0
  // CHECK: rtg.sequence_closure @seq1
  %0 = rtg.sequence_closure @seq0
  %1 = rtg.sequence_closure @seq1(%c50_i32 : i32)
  // CHECK-NOT: rtg.bag_select_random
  %bag = rtg.bag_create (%0 : %c50, %1 : %c50) : !rtg.sequence
  %seq = rtg.bag_select_random %bag : !rtg.bag<!rtg.sequence>
  rtg.invoke %seq : !rtg.sequence
}
