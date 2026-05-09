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

  // CHECK: moore.global_variable @"p::stored" : !moore.ustruct<
  vif_t stored;

  // CHECK: func.func private @"p::set"(
  // CHECK: moore.get_global_variable @"p::stored"
  // CHECK: moore.blocking_assign
  function void set
      (vif_t v);
    stored = v;
  endfunction

  // CHECK: func.func private @"p::get"(
  // CHECK: moore.get_global_variable @"p::stored"
  // CHECK: moore.read
  function vif_t get
      ();
    return stored;
  endfunction
endpackage

// CHECK-LABEL: moore.module @top
module top;
  logic clk;
  output_if out(clk);

  p::vif_t v;

  initial begin
    p::set(out);
    v = p::get();
    v.data = 8'h5a;
  end
endmodule
