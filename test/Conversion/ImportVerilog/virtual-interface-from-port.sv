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

// CHECK-LABEL: moore.module private @top(in %out_clk : !moore.ref<l1>, in %out_data : !moore.ref<l8>)
module top
    (output_if out);
  consumer c;

  initial begin
    // CHECK: [[C:%.*]] = moore.variable : <class<@consumer>>
    // CHECK: [[CNEW:%.*]] = moore.class.new
    // CHECK: moore.blocking_assign [[C]], [[CNEW]]
    c = new;

    // CHECK: [[VIF:%.*]] = moore.struct_create %out_clk, %out_data
    // CHECK: [[CVAL:%.*]] = moore.read [[C]] : <class<@consumer>>
    // CHECK: func.call @"consumer::set_vif"([[CVAL]], [[VIF]])
    c.set_vif(out);

    // CHECK: [[B:%.*]] = moore.constant 17 : l8
    // CHECK: [[CVAL2:%.*]] = moore.read [[C]] : <class<@consumer>>
    // CHECK: func.call @"consumer::drive"([[CVAL2]], [[B]])
    c.drive(8'h11);
  end
endmodule

module tb;
  logic clk;
  output_if out_if(clk);
  top t(out_if);
endmodule
