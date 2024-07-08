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

  /// *******************
  /// Data out management.
  ///

  localparam int FROM_HOST_SIZE_BYTES = int'((FROM_HOST_SIZE_BITS+7)/8);
  // The number of bits over a byte.
  localparam int FROM_HOST_SIZE_BITS_DIFF = FROM_HOST_SIZE_BITS % 8;
  localparam int FROM_HOST_SIZE_BYTES_FLOOR = int'(FROM_HOST_SIZE_BITS/8);
  localparam int FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS
      = FROM_HOST_SIZE_BYTES_FLOOR * 8;

  byte unsigned DataOutBuffer[FROM_HOST_SIZE_BYTES-1:0];
  always @(posedge clk) begin
    if (~rst) begin
      if (DataOutValid && DataOutReady) // A transfer occurred.
        DataOutValid <= 1'b0;

      if (!DataOutValid || DataOutReady) begin
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
            DataOutValid <= 1'b1;
          else if (data_limit == 0)
            begin end // No message.
          else
            $error(
              "cosim_ep_tryget(%s, *, %d -> %d) did not load entire buffer!",
              ENDPOINT_ID, FROM_HOST_SIZE_BYTES, data_limit);
        end
      end
    end else begin
      DataOutValid <= 1'b0;
    end
  end

  // Assign packed output bit array from unpacked byte array.
  genvar iOut;
  generate
    for (iOut=0; iOut<FROM_HOST_SIZE_BYTES_FLOOR; iOut++)
      assign DataOut[((iOut+1)*8)-1:iOut*8] = DataOutBuffer[iOut];
    if (FROM_HOST_SIZE_BITS_DIFF != 0)
      // If the type is not a multiple of 8, we've got to copy the extra bits.
      assign DataOut[(FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS +
                      FROM_HOST_SIZE_BITS_DIFF - 1) :
                        FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS]
             = DataOutBuffer[FROM_HOST_SIZE_BYTES - 1]
                            [FROM_HOST_SIZE_BITS_DIFF - 1 : 0];
  endgenerate

  initial begin
    $display("FROM_HOST_SIZE_BITS: %d", FROM_HOST_SIZE_BITS);
    $display("FROM_HOST_SIZE_BYTES: %d", FROM_HOST_SIZE_BYTES);
    $display("FROM_HOST_SIZE_BITS_DIFF: %d", FROM_HOST_SIZE_BITS_DIFF);
    $display("FROM_HOST_SIZE_BYTES_FLOOR: %d", FROM_HOST_SIZE_BYTES_FLOOR);
    $display("FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS: %d",
             FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS);
  end

endmodule
