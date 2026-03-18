// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

interface output_if
    (input logic clk);
  logic [7 : 0] data;
endinterface

// CHECK-LABEL: moore.module @top
module top;
  logic clk0, clk1;
  output_if out0(clk0);
  output_if out1(clk1);

  virtual output_if vifs[2];
  virtual output_if tmp;
  logic [7 : 0] s;

  initial begin
    vifs[0] = out0;
    vifs[1] = out1;

    // CHECK: moore.struct_create
    // CHECK: moore.struct_create

    tmp = vifs[1];

    // CHECK: moore.struct_extract
    // CHECK-SAME: "data"
    // CHECK: moore.blocking_assign
    tmp.data = 8'h22;

    // CHECK: moore.struct_extract
    // CHECK-SAME: "data"
    // CHECK: moore.read
    s = tmp.data;
  end
endmodule
