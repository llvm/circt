// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog %s | FileCheck %s --check-prefix=HW
// REQUIRES: slang

// Test that DPI-C imported functions are emitted as extern declarations
// (no body) and that calls to them survive through full lowering.

// --- Moore-level checks: extern declarations with no body ---

// MOORE: func.func private @void_dpi(!moore.i32)
// MOORE-NOT: return

// MOORE: func.func private @nonvoid_dpi(!moore.i32) -> !moore.i32
// MOORE-NOT: return

// MOORE: func.func private @dpi_with_output(!moore.i32, !moore.ref<i32>)
// MOORE-NOT: return

import "DPI-C" function void void_dpi(input int a);
import "DPI-C" function int nonvoid_dpi(input int a);
import "DPI-C" function void dpi_with_output(input int a, output int b);

// --- HW-level checks: declarations survive lowering with correct types ---

// HW-DAG: func.func private @void_dpi(i32)
// HW-DAG: func.func private @nonvoid_dpi(i32) -> i32
// HW-DAG: func.func private @dpi_with_output(i32, !llhd.ref<i32>)

// MOORE-LABEL: moore.module @DpiCallTest
// HW-LABEL: hw.module @DpiCallTest
module DpiCallTest(input int in_val, output int out_val);
  int result;

  // MOORE: func.call @void_dpi
  // MOORE: func.call @nonvoid_dpi
  // MOORE: func.call @dpi_with_output
  // HW: func.call @void_dpi
  // HW: func.call @nonvoid_dpi
  // HW: func.call @dpi_with_output

  always_comb begin
    void_dpi(in_val);
    result = nonvoid_dpi(in_val);
    dpi_with_output(in_val, result);
  end

  assign out_val = result;
endmodule

// --- chandle type: maps to !moore.chandle at Moore level, !llvm.ptr at HW ---

import "DPI-C" function chandle chandle_init(input int size);
import "DPI-C" function void chandle_tick(input chandle ctx, input int a);

// MOORE: func.func private @chandle_init(!moore.i32) -> !moore.chandle
// MOORE: func.func private @chandle_tick(!moore.chandle, !moore.i32)
// HW: func.func private @chandle_init(i32) -> !llvm.ptr
// HW: func.func private @chandle_tick(!llvm.ptr, i32)

// MOORE-LABEL: moore.module @ChandleTest
// HW-LABEL: hw.module @ChandleTest
module ChandleTest(input logic clock, input int in_val);
  chandle ctx;

  initial begin
    ctx = chandle_init(32);
  end

  always @(posedge clock) begin
    chandle_tick(ctx, in_val);
  end
endmodule
