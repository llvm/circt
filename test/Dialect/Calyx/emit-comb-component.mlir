// RUN: circt-translate --export-calyx --split-input-file --verify-diagnostics %s | FileCheck %s --strict-whitespace

module attributes {calyx.entrypoint = "main"} {
  // CHECK-LABEL: comb component A(in: 32) -> (out: 32) {
  calyx.comb_component @A(%in: i32) -> (%out: i32) {
    calyx.wires {
      // CHECK: out = in;
      calyx.assign %out = %in : i32
    }
  }

  // CHECK-LABEL: component main<"static"=1>(in: 32, @go go: 1, @clk clk: 1, @reset reset: 1) -> (out: 32, @done done: 1) {
  calyx.component @main(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i32, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    %A.in, %A.out = calyx.instance @A_0 of @A : i32, i32

    calyx.wires {
      // CHECK: done = 1'd1;
      calyx.assign %done = %c1_1 : i1
      // CHECK: A_0.in = in;
      calyx.assign %A.in = %in : i32
      // CHECK: out = A_0.out;
      calyx.assign %out = %A.out : i32
    }
    calyx.control {}
  } {static = 1}
}
