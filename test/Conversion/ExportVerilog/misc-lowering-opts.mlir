// RUN: circt-opt %s -export-verilog --split-input-file | FileCheck %s

!fooTy = !hw.struct<bar: i4>

module attributes {circt.loweringOptions="emitWireInPorts"} {
// CHECK-LABEL: module Foo(
// CHECK-NEXT:    input  wire                                   a,
// CHECK-NEXT:    input  wire struct packed {logic [3:0] bar; } foo,
// CHECK-NEXT:    output wire [2:0]                             x
// CHECK:       endmodule
hw.module @Foo(in %a: i1, in %foo: !fooTy, out x: i3) {
  %c0_i3 = hw.constant 0 : i3
  hw.output %c0_i3 : i3
}
}

// -----

module attributes {circt.loweringOptions="caseInsensitiveKeywords"} {
  // CHECK-LABEL: caseInsensitiveKeywords
  hw.module @caseInsensitiveKeywords() {}
  // CHECK:      module MODULE_0
  // CHECK:        input Module_0,
  // CHECK-NEXT:   output MoDuLe_0
  // CHECK:        assign MoDuLe_0 = Module_0;
  hw.module @MODULE(in %Module: i1, out MoDuLe: i1) {
    hw.output %Module : i1
  }
}
