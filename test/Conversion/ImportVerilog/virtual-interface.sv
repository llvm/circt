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

// CHECK-LABEL: moore.class.classdecl @consumer
// CHECK: moore.class.propertydecl @vif : !moore.ustruct<
// CHECK-SAME: clk:
// CHECK-SAME: ref<l1>
// CHECK-SAME: data:
// CHECK-SAME: ref<l8>
// CHECK: }
class consumer;
  virtual output_if vif;

  // CHECK: func.func private @"consumer::set_vif"(
  // CHECK-SAME: !moore.class<@consumer>
  // CHECK-SAME: !moore.ustruct<
  // CHECK: moore.class.property_ref
  // CHECK: moore.blocking_assign
  function void set_vif
      (virtual output_if v);
    vif = v;
  endfunction

  // CHECK: func.func private @"consumer::sample"(
  // CHECK: moore.struct_extract
  // CHECK-SAME: "data"
  // CHECK-SAME: -> ref<l8>
  // CHECK: moore.read
  function logic [7 : 0]
    sample();
    return vif.data;
  endfunction

  // CHECK: func.func private @"consumer::drive_byte"(
  // CHECK: moore.struct_extract
  // CHECK-SAME: "data"
  // CHECK: moore.blocking_assign
  function void drive_byte
      (logic [7 : 0] v);
    vif.data = v;
  endfunction

  // CHECK: moore.coroutine private @"consumer::wait_posedge_clk"(
  // CHECK: moore.wait_event
  // CHECK: moore.struct_extract
  // CHECK-SAME: "clk"
  task wait_posedge_clk
      ();
    @(posedge vif.clk);
  endtask
endclass

// CHECK-LABEL: moore.module @top
module top;
  logic clk;
  output_if out(clk);

  consumer c;
  logic [7 : 0] s;

  initial begin
    c = new;
    c.set_vif(out);
    c.drive_byte(8'hA5);
    s = c.sample();
    c.wait_posedge_clk();
  end
endmodule
