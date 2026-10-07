// RUN: circt-opt %s -export-verilog | FileCheck %s --strict-whitespace

// CHECK-LABEL: module Scalars(
// CHECK-NEXT:   input  [3:0] a,
// CHECK-NEXT:   inout  [3:0] n,
// CHECK-NEXT:   output [3:0] o
// CHECK-NEXT: );
hw.module @Scalars(in %a: i4, in %n: !sv.net<i4>, out o: i4) {
  hw.output %a : i4
}

// CHECK-LABEL: module StructElement(
// CHECK-NEXT:   inout struct packed {logic [3:0] f; } n
// CHECK-NEXT: );
hw.module @StructElement(in %n: !sv.net<!hw.struct<f: i4>>) {
}

// CHECK-LABEL: module NoShareAcrossKind(
// CHECK-NEXT:   input [3:0] a,
// CHECK-NEXT:   inout [3:0] n
// CHECK-NEXT: );
hw.module @NoShareAcrossKind(in %a: i4, in %n: !sv.net<i4>) {
}
