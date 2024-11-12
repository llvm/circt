// RUN: circt-opt --rtg-elaborate=seed=0 %s | FileCheck %s --check-prefixes=CHECK-SEED0,CHECK
// RUN: circt-opt --rtg-elaborate=seed=1 %s | FileCheck %s --check-prefixes=CHECK-SEED1,CHECK

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

// CHECK-LABEL: rtg.test @seq2
rtg.test @seq2 : !rtg.target<> {
  // CHECK-SEED0: [[C50:%.+]] = arith.constant 50 : i32
  %c50 = arith.constant 50 : i32
  // CHECK-NOT: rtg.select_random
  // CHECK-SEED0: rtgtest.instr_a %{{.*}}, [[C50]]
  // CHECK-SEED1: rtg.label.decl
  %0 = rtg.sequence_closure @seq0
  %1 = rtg.sequence_closure @seq1(%c50 : i32)
  // CHECK-NOT: rtg.select_random
  rtg.select_random [%0, %1]((), () : (), ()), [%c50, %c50] : !rtg.sequence, !rtg.sequence
}
