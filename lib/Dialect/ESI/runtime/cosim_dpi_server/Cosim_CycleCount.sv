//===- Cosim_CycleCount.sv - ESI cycle counter module -------*- verilog -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A cosimulated design should instantiate this ONCE to provide cycle count
// information to the ESI runtime. The module maintains a 64-bit cycle counter
// and exports a DPI function that the C++ runtime can call to query it.
//
//===----------------------------------------------------------------------===//

import Cosim_DpiPkg::*;

module Cosim_CycleCount
#(
  // Core clock frequency in Hz. Used by the runtime to convert cycles to time.
  parameter longint unsigned CORE_CLOCK_FREQUENCY_HZ = 100_000_000
)(
  input logic clk,
  input logic rst
);

  // 64-bit cycle counter.
  longint unsigned cycle_count;

  // DPI export function - allows C++ to call into SV to get the cycle count.
  export "DPI-C" function c2svCosimserverGetCycleCount;
  function automatic longint unsigned c2svCosimserverGetCycleCount();
    return cycle_count;
  endfunction

  // Cycle counter logic.
  always_ff @(posedge clk) begin
    if (rst)
      cycle_count <= 0;
    else
      cycle_count <= cycle_count + 1;
  end

  // Register the cycle count callback with the cosim server at initialization.
  initial begin
    // Small delay to ensure the server is initialized.
    #1;
    cosim_set_cycle_count_callback(CORE_CLOCK_FREQUENCY_HZ);
  end

endmodule
