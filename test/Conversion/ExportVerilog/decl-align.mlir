// RUN: circt-opt %s -export-verilog | FileCheck %s --strict-whitespace

sv.macro.decl @foo

// CHECK-LABEL: module Decl
hw.module @Decl() {
  // CHECK: wire [3:0] x;
  %x = sv.wire : !sv.net<i4>
  // CHECK: wire       y;
  %y = sv.wire : !sv.net<i1>
  sv.ifdef @foo {
    // CHECK: wire [11:0][9:0][3:0] w;
    %w = sv.wire : !sv.net<!hw.array<12 x array<10xi4>>>
  }
  // CHECK: wire [5:0] z;
  %z = sv.wire : !sv.net<i6>
}
