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
// and uses request/response channels to provide the count to the runtime.
//
//===----------------------------------------------------------------------===//

import Cosim_DpiPkg::*;

module Cosim_CycleCount
#(
  // Core clock frequency in Hz. Used by the runtime to convert cycles to time.
  parameter longint unsigned CORE_CLOCK_FREQUENCY_HZ = 0
)(
  input logic clk,
  input logic rst
);

  // 64-bit cycle counter.
  longint unsigned cycle_count;

  always_ff @(posedge clk) begin
    if (rst)
      cycle_count <= 0;
    else
      cycle_count <= cycle_count + 1;
  end

  // Request channel (From Host)
  // 0-bit message used as a trigger.
  wire req_valid;
  wire req_ready;

  Cosim_Endpoint_FromHost #(
    .ENDPOINT_ID("__cosim_cycle_count.arg"),
    .FROM_HOST_TYPE_ID("i1"),
    .FROM_HOST_SIZE_BITS(1)
  ) req_ep (
    .clk(clk),
    .rst(rst),
    .DataOutValid(req_valid),
    .DataOutReady(req_ready),
    .DataOut()
  );

  // Response channel (To Host)
  typedef struct packed {
    longint unsigned cycle;
    longint unsigned freq;
  } RespStruct;

  wire resp_ready;
  wire resp_valid;
  RespStruct resp_data;

  assign resp_data.cycle = cycle_count;
  assign resp_data.freq = CORE_CLOCK_FREQUENCY_HZ;

  // Handshake logic:
  // When request arrives (req_valid), we are ready to send response (resp_valid).
  // We consume request only when response is accepted.
  assign resp_valid = req_valid;
  assign req_ready = resp_ready;

  Cosim_Endpoint_ToHost #(
    .ENDPOINT_ID("__cosim_cycle_count.result"),
    .TO_HOST_TYPE_ID("struct{cycle:int<64>,freq:int<64>}"),
    .TO_HOST_SIZE_BITS(128)
  ) resp_ep (
    .clk(clk),
    .rst(rst),
    .DataInValid(resp_valid),
    .DataInReady(resp_ready),
    .DataIn(resp_data)
  );

endmodule
