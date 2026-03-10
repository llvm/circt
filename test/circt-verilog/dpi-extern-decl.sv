// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Test that DPI-C imported functions are emitted as extern declarations

// --- HW-level checks: declarations survive lowering with correct types ---

// CHECK-DAG: func.func private @void_dpi(i32)
// CHECK-DAG: func.func private @nonvoid_dpi(i32) -> i32
// CHECK-DAG: func.func private @dpi_with_output(i32, !llhd.ref<i32>)

import "DPI-C" function void void_dpi(input int a);
import "DPI-C" function int nonvoid_dpi(input int a);
import "DPI-C" function void dpi_with_output(input int a, output int b);

// CHECK-LABEL: hw.module @DpiCallTest
module DpiCallTest(input int in_val, output int out_val);
  int result;

  // CHECK: func.call @void_dpi
  // CHECK: func.call @nonvoid_dpi
  // CHECK: func.call @dpi_with_output

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

// CHECK: func.func private @chandle_init(i32) -> !llvm.ptr
// CHECK: func.func private @chandle_tick(!llvm.ptr, i32)

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
