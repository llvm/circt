// RUN: circt-translate --export-calyx --verify-diagnostics %s | FileCheck %s --strict-whitespace

calyx.program {
  // CHECK-LABEL: component A(in: 8, go: 1, clk: 1, reset: 1) -> (out: 8, done: 1) {
  calyx.component @A(%in: i8, %go: i1, %clk: i1, %reset: i1) -> (%out: i8, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: component main(go: 1, clk: 1, reset: 1) -> (done: 1) {
  calyx.component @main(%go: i1, %clk: i1, %reset: i1) -> (%done: i1) {
    // CHECK: cells {
    // CHECK-NEXT:   c0 = A();
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.cell "c0" @A : i8, i1, i1, i1, i8, i1

    // CHECK: wires {
    calyx.wires {
      // CHECK: group Group1 {
      // CHECK:   c0.in = c0.out;
      calyx.group @Group1 {
        calyx.assign %c0.in = %c0.out : i8
        calyx.group_done %c0.done : i1
      }
      // CHECK:   c0.go = 1'd0;
      %c1 = hw.constant 0 : i1
      calyx.assign %c0.go = %c1 : i1
    }
    // CHECK: control {
    calyx.control {
      calyx.seq {
        calyx.enable @Group1
      }
    }
  }
}
