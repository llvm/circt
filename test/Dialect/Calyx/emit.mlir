// RUN: circt-translate --export-calyx --verify-diagnostics %s | FileCheck %s --strict-whitespace

calyx.program {
  // CHECK-LABEL: component A(in: 8, go: 1, clk: 1, reset: 1) -> (out: 8, done: 1) {
  calyx.component @A(%in: i8, %go: i1, %clk: i1, %reset: i1) -> (%out: i8, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: component main(go: 1, clk: 1, reset: 1) -> (done: 1) {
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    // CHECK-NEXT: cells {
    // CHECK-NEXT:   c0 = A();
    // CHECK-NEXT: }
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.cell "c0" @A : i8, i1, i1, i1, i8, i1

    // CHECK-NEXT: wires {
    // CHECK-NEXT:   group Group1 {
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    calyx.wires {
      calyx.group @Group1 {
        calyx.group_done %c0.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.enable @Group1
      }
    }
  }
}