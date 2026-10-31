// RUN: circt-opt %s -export-verilog --split-input-file | FileCheck %s --strict-whitespace 

!fooTy = !hw.struct<bar: i4>

module attributes {circt.loweringOptions="svPortDeclStyle=wire"} {
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

// alwaysEmitBeginEnd forces begin/end even for single-statement bodies.
module attributes {circt.loweringOptions="alwaysEmitBeginEnd"} {
// CHECK-LABEL: module AlwaysBeginEnd(
hw.module @AlwaysBeginEnd(in %cond: i1, in %fd: i32, in %clock: i1) {
  sv.always posedge %clock {
    // CHECK:      always @(posedge clock) begin
    // CHECK-NEXT:   if (cond) begin
    // CHECK-NEXT:     $fwrite
    // CHECK-NEXT:   end
    // CHECK-NEXT: end
    sv.if %cond {
      sv.fwrite %fd, "single\n"
    }
  }
}
}

// -----

module attributes {circt.loweringOptions="caseInsensitiveKeywords"} {
  // CHECK-LABEL: caseInsensitiveKeywords
  hw.module @caseInsensitiveKeywords() {}
  // CHECK:      module MODULE_0
  // CHECK:        input  Module_0,
  // CHECK-NEXT:   output MoDuLe_0
  // CHECK:        assign MoDuLe_0 = Module_0;
  hw.module @MODULE(in %Module: i1, out MoDuLe: i1) {
    hw.output %Module : i1
  }
}

// -----

!barTy = !hw.struct<bar: i4>

module attributes {circt.loweringOptions="svPortDeclStyle=varLogic"} {
// CHECK-LABEL: module Bar(
// CHECK-NEXT:    input  var logic                                   a,
// CHECK-NEXT:    input  var logic [3:0]                             b,
// CHECK-NEXT:    input  struct packed {logic [3:0] bar; } bar,
// CHECK-NEXT:    output var logic [2:0]                             c
// CHECK:       endmodule
hw.module @Bar(in %a: i1, in %b: i4, in %bar: !barTy, out c: i3) {
  %c0_i3 = hw.constant 0 : i3
  hw.output %c0_i3 : i3
}
}

// -----

module attributes {circt.loweringOptions="svPortDeclStyle=varLogic"} {
// CHECK-LABEL: module Grouped(
// CHECK-NEXT:    input  var logic [3:0] p,
// CHECK-NEXT:                           q,
// CHECK-NEXT:    output var logic [2:0] x
// CHECK:       endmodule
hw.module @Grouped(in %p: i4, in %q: i4, out x: i3) {
  %c0_i3 = hw.constant 0 : i3
  hw.output %c0_i3 : i3
}
}

// -----

module attributes {circt.loweringOptions="disallowReplicate"} {
// CHECK-LABEL: module ReplicateOptOut(
hw.module @ReplicateOptOut(in %arg0: i7, in %arg1: i1, out r1: i21, out r2: i4, out r3: i16) {
  // CHECK: assign r1 = {arg0, arg0, arg0};
  %r1 = comb.replicate %arg0 : (i7) -> i21
  // CHECK: assign r2 = {arg1, arg1, arg1, arg1};
  %r2 = comb.replicate %arg1 : (i1) -> i4
  // CHECK: assign r3 = {{arg0, arg1}, {arg0, arg1}};
  %c = comb.concat %arg0, %arg1 : i7, i1
  %r3 = comb.replicate %c : (i8) -> i16
  hw.output %r1, %r2, %r3 : i21, i4, i16
}
}

