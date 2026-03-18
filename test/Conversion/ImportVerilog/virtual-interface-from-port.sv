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

class consumer;
  virtual output_if vif;

  function void set_vif
      (virtual output_if v);
    vif = v;
  endfunction

  function void drive
      (logic [7 : 0] v);
    vif.data = v;
  endfunction
endclass

// CHECK-LABEL: moore.module{{.*}} @top
// CHECK: moore.struct_create
module top
    (output_if out);
  consumer c;

  initial begin
    c = new;
    c.set_vif(out);
    c.drive(8'h11);
  end
endmodule

module tb;
  logic clk;
  output_if out_if(clk);
  top t(out_if);
endmodule
