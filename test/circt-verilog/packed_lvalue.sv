// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: hw.module @Foo(
module Foo(input logic [1023:0] x);
  logic [7:0][63:0] a, b;
  always_comb {a, b} = x;
endmodule