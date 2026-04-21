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

package p;
  typedef virtual output_if vif_t;

  // CHECK-LABEL: moore.class.classdecl @"p::consumer"
  // CHECK: moore.class.propertydecl @vif : !moore.ustruct<
  // CHECK-SAME: data:
  class consumer;
    vif_t vif;

    function void set_vif
        (vif_t v);
      vif = v;
    endfunction

    // CHECK: func.func private @"p::consumer::sample"
    // CHECK: moore.struct_extract
    // CHECK-SAME: "data"
    function logic [7 : 0]
      sample();
      return vif.data;
    endfunction

    function void drive_byte
        (logic [7 : 0] v);
      vif.data = v;
    endfunction
  endclass
endpackage

module top;
  logic clk;
  output_if out(clk);
  p::consumer c;
  logic [7 : 0] s;

  initial begin
    c = new;
    c.set_vif(out);
    c.drive_byte(8'h5a);
    s = c.sample();
  end
endmodule
