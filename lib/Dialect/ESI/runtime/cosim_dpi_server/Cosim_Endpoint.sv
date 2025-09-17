//===- Cosim_Endpoint.sv - ESI cosim primary RTL module -----*- verilog -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Package: Cosim_DpiPkg
//
// Main cosim <--> dpi bridge module.


//
//===----------------------------------------------------------------------===//

import Cosim_DpiPkg::*;

module Cosim_Endpoint_ToHost
#(
  parameter string ENDPOINT_ID = "",
  parameter string TO_HOST_TYPE_ID = "",
  parameter int TO_HOST_SIZE_BITS = -1
)
(
  input  logic clk,
  input  logic rst,

  input  logic DataInValid,
  output logic DataInReady,
  input  logic [TO_HOST_SIZE_BITS-1:0] DataIn
);

  initial begin
    int rc;
    rc = cosim_init();
    if (rc != 0)
      $error("Cosim init failed (%d)", rc);
    rc = cosim_ep_register(ENDPOINT_ID, "", 0,
                            TO_HOST_TYPE_ID, TO_HOST_SIZE_BYTES);
    if (rc != 0)
      $error("Cosim endpoint (%s) register failed: %d", ENDPOINT_ID, rc);
  end

  /// **********************
  /// Data in management.
  ///

  localparam int TO_HOST_SIZE_BYTES = int'((TO_HOST_SIZE_BITS+7)/8);
  // The number of bits over a byte.
  localparam int TO_HOST_SIZE_BITS_DIFF = TO_HOST_SIZE_BITS % 8;
  localparam int TO_HOST_SIZE_BYTES_FLOOR = int'(TO_HOST_SIZE_BITS/8);
  localparam int TO_HOST_SIZE_BYTES_FLOOR_IN_BITS
      = TO_HOST_SIZE_BYTES_FLOOR * 8;

  assign DataInReady = 1'b1;
  byte unsigned DataInBuffer[TO_HOST_SIZE_BYTES-1:0];

  always@(posedge clk) begin
    if (~rst) begin
      if (DataInValid) begin
        int rc;
        rc = cosim_ep_tryput(ENDPOINT_ID, DataInBuffer, TO_HOST_SIZE_BYTES);
        if (rc != 0)
          $error("cosim_ep_tryput(%s, *, %d) = %d Error! (Data lost)",
            ENDPOINT_ID, TO_HOST_SIZE_BYTES, rc);
      end
    end
  end

  // Assign packed input bit array to unpacked byte array.
  genvar iIn;
  generate
    for (iIn=0; iIn<TO_HOST_SIZE_BYTES_FLOOR; iIn++)
      assign DataInBuffer[iIn] = DataIn[((iIn+1)*8)-1:iIn*8];
    if (TO_HOST_SIZE_BITS_DIFF != 0)
      // If the type is not a multiple of 8, we've got to copy the extra bits.
      assign DataInBuffer[TO_HOST_SIZE_BYTES - 1]
                         [TO_HOST_SIZE_BITS_DIFF - 1:0] =
             DataIn[(TO_HOST_SIZE_BYTES_FLOOR_IN_BITS +
                     TO_HOST_SIZE_BITS_DIFF - 1) :
                       TO_HOST_SIZE_BYTES_FLOOR_IN_BITS];
  endgenerate

  initial begin
    $display("TO_HOST_SIZE_BITS: %d", TO_HOST_SIZE_BITS);
    $display("TO_HOST_SIZE_BYTES: %d", TO_HOST_SIZE_BYTES);
    $display("TO_HOST_SIZE_BITS_DIFF: %d", TO_HOST_SIZE_BITS_DIFF);
    $display("TO_HOST_SIZE_BYTES_FLOOR: %d", TO_HOST_SIZE_BYTES_FLOOR);
    $display("TO_HOST_SIZE_BYTES_FLOOR_IN_BITS: %d",
             TO_HOST_SIZE_BYTES_FLOOR_IN_BITS);
  end

endmodule

module Cosim_Endpoint_FromHost
#(
  parameter string ENDPOINT_ID = "",
  parameter string FROM_HOST_TYPE_ID = "",
  parameter int FROM_HOST_SIZE_BITS = -1
)
(
  input  logic clk,
  input  logic rst,

  output logic DataOutValid,
  input  logic DataOutReady,
  output logic [FROM_HOST_SIZE_BITS-1:0] DataOut
);

  // Handle initialization logic.
  initial begin
    int rc;
    rc = cosim_init();
    if (rc != 0)
      $error("Cosim init failed (%d)", rc);
    rc = cosim_ep_register(ENDPOINT_ID, FROM_HOST_TYPE_ID, FROM_HOST_SIZE_BYTES,
                            "", 0);
    if (rc != 0)
      $error("Cosim endpoint (%s) register failed: %d", ENDPOINT_ID, rc);
  end

  /// ***********************
  /// Useful constants.

  localparam int FROM_HOST_SIZE_BYTES = int'((FROM_HOST_SIZE_BITS+7)/8);
  // The number of bits over a byte.
  localparam int FROM_HOST_SIZE_BITS_DIFF = FROM_HOST_SIZE_BITS % 8;
  localparam int FROM_HOST_SIZE_BYTES_FLOOR = int'(FROM_HOST_SIZE_BITS/8);
  localparam int FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS
      = FROM_HOST_SIZE_BYTES_FLOOR * 8;

  // Buffer to hold incoming data directly from the DPI calls.
  byte unsigned DataOutBuffer[FROM_HOST_SIZE_BYTES-1:0];

  // Pipeline interface signals for buffering DataOut.
  // DataOut_a_valid is asserted when a complete message is available in
  // DataOut_a buffer and waiting to be accepted by the skid buffer.
  logic DataOut_a_valid;
  // Packed representation of the byte buffer to feed the skid input.
  logic [FROM_HOST_SIZE_BITS-1:0] DataOut_a;
  // Ready/valid wires between this module and the skid buffer.
  wire DataOut_a_ready;
  wire DataOut_x_valid;
  wire [FROM_HOST_SIZE_BITS-1:0] DataOut_x;

  always @(posedge clk) begin
    if (~rst) begin
      // If the skid buffer accepted the input token this cycle, clear the
      // internal valid that indicates we have buffered data.
      if (DataOut_a_valid && DataOut_a_ready)
        DataOut_a_valid <= 1'b0;

      // Only attempt to fetch data from the host when the skid buffer can
      // accept it (DataOut_a_ready).
      if (!DataOut_a_valid && DataOut_a_ready) begin
        int data_limit;
        int rc;

        data_limit = FROM_HOST_SIZE_BYTES;
        rc = cosim_ep_tryget(ENDPOINT_ID, DataOutBuffer, data_limit);
        if (rc < 0) begin
          $error("cosim_ep_tryget(%s, *, %d -> %d) returned an error (%d)",
            ENDPOINT_ID, FROM_HOST_SIZE_BYTES, data_limit, rc);
        end else if (rc > 0) begin
          $error("cosim_ep_tryget(%s, *, %d -> %d) had data left over! (%d)",
            ENDPOINT_ID, FROM_HOST_SIZE_BYTES, data_limit, rc);
        end else if (rc == 0) begin
          if (data_limit == FROM_HOST_SIZE_BYTES)
            DataOut_a_valid <= 1'b1;
          else if (data_limit == 0)
            begin end // No message.
          else
            $error(
              "cosim_ep_tryget(%s, *, %d -> %d) did not load entire buffer!",
              ENDPOINT_ID, FROM_HOST_SIZE_BYTES, data_limit);
        end
      end
    end else begin
      DataOut_a_valid <= 1'b0;
    end
  end

  // Pack the byte array into a single vector that will be handed to the
  // skid buffer as its input payload.
  genvar iOut;
  generate
    for (iOut=0; iOut<FROM_HOST_SIZE_BYTES_FLOOR; iOut++)
      assign DataOut_a[((iOut+1)*8)-1:iOut*8] = DataOutBuffer[iOut];
    if (FROM_HOST_SIZE_BITS_DIFF != 0)
      // If the type is not a multiple of 8, copy the extra bits.
      assign DataOut_a[(FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS +
                        FROM_HOST_SIZE_BITS_DIFF - 1) :
                          FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS]
             = DataOutBuffer[FROM_HOST_SIZE_BYTES - 1]
                            [FROM_HOST_SIZE_BITS_DIFF - 1 : 0];
  endgenerate

  /// *******************
  /// Data out management.
  ///
  /// It has been observed that some simulators (verilator) does not handle
  /// combinational driving from DPI well (resulting in non-determinism wrt
  /// some of the combinational outputs being dropped/replicated). Questa does
  /// not show this behavior.
  /// A mitigation is to add a skid buffer to decouple the DPI interface
  /// from the output interface.

  // Instantiate a single-stage pipeline to buffer tokens that arrive from
  // the host and present them to the module's output with proper
  // ready/valid handshaking.
  ESI_PipelineStage #(.WIDTH(FROM_HOST_SIZE_BITS)) out_pipe (
    .clk(clk),
    .rst(rst),
    .a_valid(DataOut_a_valid),
    .a(DataOut_a),
    .a_ready(DataOut_a_ready),
    .x_valid(DataOut_x_valid),
    .x(DataOut_x),
    .x_ready(DataOutReady)
  );

  // Drive the module's outward-facing signals from the pipeline output.
  assign DataOutValid = DataOut_x_valid;
  assign DataOut = DataOut_x;

  initial begin
    $display("FROM_HOST_SIZE_BITS: %d", FROM_HOST_SIZE_BITS);
    $display("FROM_HOST_SIZE_BYTES: %d", FROM_HOST_SIZE_BYTES);
    $display("FROM_HOST_SIZE_BITS_DIFF: %d", FROM_HOST_SIZE_BITS_DIFF);
    $display("FROM_HOST_SIZE_BYTES_FLOOR: %d", FROM_HOST_SIZE_BYTES_FLOOR);
    $display("FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS: %d",
             FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS);
  end

endmodule
