// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Test that DPI-C imported functions are emitted as extern declarations

// --- HW-level checks: declarations survive lowering with correct types ---

// CHECK-DAG: sim.func.dpi private @void_dpi(in %a : i32)
// CHECK-DAG: sim.func.dpi private @nonvoid_dpi(in %a : i32, return return : i32)
// CHECK-DAG: sim.func.dpi private @dpi_with_output(in %a : i32, out b : i32)

import "DPI-C" function void void_dpi(input int a);
import "DPI-C" function int nonvoid_dpi(input int a);
import "DPI-C" function void dpi_with_output(input int a, output int b);

// CHECK-LABEL: hw.module @DpiCallTest
module DpiCallTest(input int in_val, output int out_val);
  int result;

  // CHECK: sim.func.dpi.call @void_dpi
  // CHECK: sim.func.dpi.call @nonvoid_dpi
  // CHECK: sim.func.dpi.call @dpi_with_output

  always_comb begin
    void_dpi(in_val);
    result = nonvoid_dpi(in_val);
    dpi_with_output(in_val, result);
  end

  assign out_val = result;
endmodule

// --- chandle type: maps to !llvm.ptr at HW ---

import "DPI-C" function chandle chandle_init(input int size);
import "DPI-C" function void chandle_tick(input chandle ctx, input int a);

// CHECK: sim.func.dpi private @chandle_init(in %size : i32, return return : !llvm.ptr)
// CHECK: sim.func.dpi private @chandle_tick(in %ctx : !llvm.ptr, in %a : i32)

// CHECK-LABEL: hw.module @ChandleTest
module ChandleTest(input logic clock, input int in_val);
  chandle ctx;

  initial begin
    ctx = chandle_init(32);
  end

  always @(posedge clock) begin
    chandle_tick(ctx, in_val);
  end
endmodule
