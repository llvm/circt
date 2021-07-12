// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK: calyx.program {
calyx.program {
  // CHECK-LABEL: calyx.component @A(%in: i8, %go: i1, %clk: i1, %reset: i1) -> (%out: i8, %done: i1) {
  calyx.component @A(%in: i8, %go: i1, %clk: i1, %reset: i1) -> (%out: i8, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @B (%go: i1, %clk: i1, %reset: i1) -> (%out: i1, %done: i1) {
  calyx.component @B (%go: i1, %clk: i1, %reset: i1) -> (%out: i1, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    %in1, %out1 = calyx.cell "c0" @A : i8, i8
    %in2, %out2 = calyx.cell "c1" @A : i8, i8
    %out3 = calyx.cell "c2" @B : i1
    %c1_i1 = constant 1 : i1

    calyx.wires {
      // CHECK: calyx.group @Group1 {
      calyx.group @Group1 {
        // CHECK: calyx.assign %1#0 = %0#1 : i8
        // CHECK-NEXT: %3 = calyx.group_done %true : i1
        calyx.assign %in2 = %out1 : i8
        %d0 = calyx.group_done %c1_i1 : i1
      }
      calyx.group @Group2 {
        // CHECK:  calyx.assign %1#0 = %0#1, %2 ? : i8
        calyx.assign %in2 = %out1, %out3 ?  : i8

        // CHECK: %4 = calyx.group_done %true, %3 ? : i1
        %guard = comb.and %c1_i1, %out3 : i1
        %d1 = calyx.group_done %c1_i1, %guard ? : i1
      }
    }
    calyx.control {
      // CHECK:      calyx.seq {
      // CHECK-NEXT: calyx.enable @Group1
      // CHECK-NEXT: calyx.enable @Group2
      // CHECK-NEXT: calyx.seq {
      // CHECK-NEXT: calyx.enable @Group1
      calyx.seq {
        calyx.enable @Group1
        calyx.enable @Group2
        calyx.seq {
          calyx.enable @Group1
        }
      }
    }
  }
}
