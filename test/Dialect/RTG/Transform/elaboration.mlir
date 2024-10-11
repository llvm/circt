// RUN: circt-opt --rtg-elaborate=seed=0 %s | FileCheck %s --check-prefixes=CHECK-SEED0,CHECK
// RUN: circt-opt --rtg-elaborate=seed=1 %s | FileCheck %s --check-prefixes=CHECK-SEED1,CHECK

// CHECK-LABEL: rtg.sequence
%0 = rtg.sequence {
  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  %3 = rtg.label.decl "label_{0}_{1}", %1, %2 : i32, i32 -> i32
  rtg.label %3 : i32
} -> !rtg.sequence

// CHECK-LABEL: rtg.sequence
%1 = rtg.sequence {
^bb0(%arg0: i32):
  %3 = arith.constant 3 : i32
  rtgtest.instr_a %3, %arg0
} -> !rtg.sequence<i32>

// CHECK-LABEL: rtg.sequence
rtg.sequence {
  // CHECK-SEED0: [[C50:%.+]] = arith.constant 50 : i32
  %c50 = arith.constant 50 : i32
  // CHECK-NOT: rtg.select-random
  // CHECK-SEED0: rtgtest.instr_a %{{.*}}, [[C50]]
  // CHECK-SEED1: rtg.label.decl
  // CHECK-NOT: rtg.select-random
  rtg.select_random [%0, %1]((), (%c50) : (), (i32)), [%c50, %c50] : !rtg.sequence, !rtg.sequence<i32>
} -> !rtg.sequence
