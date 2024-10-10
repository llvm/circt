// RUN: circt-opt --rtg-elaborate=seed=0 %s | FileCheck %s --check-prefixes=CHECK-SEED0,CHECK
// RUN: circt-opt --rtg-elaborate=seed=1 %s | FileCheck %s --check-prefixes=CHECK-SEED1,CHECK

// CHECK-LABEL: rtg.snippet
%0 = rtg.snippet {
  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  %3 = rtg.label.decl "label_{0}_{1}", %1, %2 : i32, i32 -> i32
  rtg.label %3 : i32
}

// CHECK-LABEL: rtg.snippet
%1 = rtg.snippet {
  %3 = arith.constant 3 : i32
  %4 = arith.constant 4 : i32
  rtgtest.instr_a %3, %4
}

// CHECK-LABEL: rtg.snippet
rtg.snippet {
  %c50 = arith.constant 50 : i32
  // CHECK-NOT: rtg.select-random
  // CHECK-SEED0: rtgtest.instr_a
  // CHECK-SEED1: rtg.label.decl
  // CHECK-NOT: rtg.select-random
  rtg.select_random [%0, %1], [%c50, %c50]
}
