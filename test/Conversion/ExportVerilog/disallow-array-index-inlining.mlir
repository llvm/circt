// RUN: circt-opt --export-verilog %s | FileCheck %s
// RUN: circt-opt --test-apply-lowering-options='options=disallowArrayIndexInlining' --export-verilog %s | FileCheck %s --check-prefix=DISALLOW

// CHECK-LABEL: module Foo(
// DISALLOW-LABEL: module Foo(
hw.module @Foo(%a: !hw.array<16xi1>, %b : i4) -> (x: i1, y: i1) {
  // CHECK: assign x = a[b + 4'h1];
  // DISALLOW-DAG: wire [3:0] [[IDX0:.+]] = b + 4'h1;
  // DISALLOW-DAG: assign x = a[[[IDX0]]];
  %c1_i4 = hw.constant 1 : i4
  %0 = comb.add %b, %c1_i4 : i4
  %1 = hw.array_get %a[%0] : !hw.array<16xi1>, i4

  // CHECK: assign y = a[b * (b + 4'h2)];
  // DISALLOW-DAG: wire [3:0] [[IDX1:.+]] = b * (b + 4'h2);
  // DISALLOW-DAG: assign y = a[[[IDX1]]];
  %c2_i4 = hw.constant 2 : i4
  %2 = comb.add %b, %c2_i4 : i4
  %3 = comb.mul %b, %2 : i4
  %4 = hw.array_get %a[%3] : !hw.array<16xi1>, i4

  hw.output %1, %4 : i1, i1
}
