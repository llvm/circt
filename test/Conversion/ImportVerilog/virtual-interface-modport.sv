// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
//
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

interface output_if
    (input logic clk);
  logic [7 : 0] data;
  logic [7 : 0] data2;
  modport mp(input clk,
                   output.data(data2));
endinterface

// CHECK-LABEL: moore.class.classdecl @consumer
// CHECK: moore.class.propertydecl @vif : !moore.ustruct<
// CHECK-SAME: data:
class consumer;
  virtual output_if.mp vif;

  function void set_vif
      (virtual output_if.mp v);
    vif = v;
  endfunction

  // CHECK: func.func private @"consumer::drive"
  // CHECK: moore.struct_extract
  // CHECK-SAME: "data"
  // CHECK: moore.blocking_assign
  function void drive
      (logic [7 : 0] v);
    vif.data = v;
  endfunction
endclass

module top;
  logic clk;
  output_if out(clk);
  consumer c;

  initial begin
    c = new;
    c.set_vif(out);
    c.drive(8'h5a);
  end
endmodule
