// RUN: circt-opt %s -export-verilog --split-input-file | FileCheck %s --match-full-lines

module attributes {circt.loweringOptions = "disallowPortDeclSharing"}{
// CHECK: module Foo( // dummy:1:1
// CHECK-NEXT:  input        a,
// CHECK-NEXT:  input        b,
// CHECK-NEXT:  output [1:0] a_0,
// CHECK-NEXT:  output [1:0] b_0);
// CHECK: endmodule
hw.module @Foo(%a: i1, %b: i1) -> (a: i2, b: i2) {
  %ao = comb.concat %a, %b: i1, i1
  %bo = comb.concat %a, %a: i1, i1
  hw.output %ao, %bo : i2, i2
} loc("dummy":1:1)
}

// -----

module {
// CHECK: module Foo( // dummy:1:1
// CHECK-NEXT:  input        a,
// CHECK-NEXT:               b,
// CHECK-NEXT:  output [1:0] a_0,
// CHECK-NEXT:               b_0);
// CHECK: endmodule
hw.module @Foo(%a: i1, %b: i1) -> (a: i2, b: i2) {
  %ao = comb.concat %a, %b: i1, i1
  %bo = comb.concat %a, %a: i1, i1
  hw.output %ao, %bo : i2, i2
} loc("dummy":1:1)
}
