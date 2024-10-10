// RUN: circt-opt --rtg-elaborate %s | FileCheck %s

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
  rtg.select_random [%0, %1], [%c50, %c50]
}
