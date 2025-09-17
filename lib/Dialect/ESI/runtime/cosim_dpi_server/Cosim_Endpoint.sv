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
  /// It has been observed that some simulators (verilator) does not handle
  /// combinational driving from DPI well (resulting in non-determinism wrt.
  /// some of the combinational outputs being dropped/replicated). Questa does
  /// not show this behavior.
  /// A mitigation is to add a skid buffer to decouple the DPI interface
  /// from the output interface.
  /// *******************

  localparam int FROM_HOST_SIZE_BYTES = int'((FROM_HOST_SIZE_BITS+7)/8);
  // The number of bits over a byte.
  localparam int FROM_HOST_SIZE_BITS_DIFF = FROM_HOST_SIZE_BITS % 8;
  localparam int FROM_HOST_SIZE_BYTES_FLOOR = int'(FROM_HOST_SIZE_BITS/8);
  localparam int FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS
      = FROM_HOST_SIZE_BYTES_FLOOR * 8;

  // Skid buffer internal signals
  byte unsigned DataOutBuffer[FROM_HOST_SIZE_BYTES-1:0];
  logic internal_valid;
  logic [FROM_HOST_SIZE_BITS-1:0] internal_data;
  
  // Skid buffer registers
  logic skid_valid;
  logic [FROM_HOST_SIZE_BITS-1:0] skid_data;
  
  // Internal ready signal for DPI interface
  logic internal_ready;
  
  // Skid buffer logic
  assign internal_ready = !skid_valid || DataOutReady;
  assign DataOutValid = internal_valid || skid_valid;
  assign DataOut = skid_valid ? skid_data : internal_data;

  always @(posedge clk) begin
    if (~rst) begin
      // Skid buffer management
      if (internal_valid && internal_ready) begin
        if (!DataOutReady && !skid_valid) begin
          // Store data in skid buffer when downstream not ready
          skid_valid <= 1'b1;
          skid_data <= internal_data;
        end
      end
      
      if (DataOutReady && skid_valid) begin
        // Clear skid buffer when downstream accepts data
        skid_valid <= 1'b0;
      end
      
      // DPI interface logic
      if (internal_valid && internal_ready) // A transfer occurred.
        internal_valid <= 1'b0;

      if (!internal_valid || internal_ready) begin
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
            internal_valid <= 1'b1;
          else if (data_limit == 0)
            begin end // No message.
          else
            $error(
              "cosim_ep_tryget(%s, *, %d -> %d) did not load entire buffer!",
              ENDPOINT_ID, FROM_HOST_SIZE_BYTES, data_limit);
        end
      end
    end else begin
      internal_valid <= 1'b0;
      skid_valid <= 1'b0;
    end
  end

  // Assign packed output bit array from unpacked byte array to internal signal
  genvar iOut;
  generate
    for (iOut=0; iOut<FROM_HOST_SIZE_BYTES_FLOOR; iOut++)
      assign internal_data[((iOut+1)*8)-1:iOut*8] = DataOutBuffer[iOut];
    if (FROM_HOST_SIZE_BITS_DIFF != 0)
      // If the type is not a multiple of 8, we've got to copy the extra bits.
      assign internal_data[(FROM_HOST_SIZE_BYTES_FLOOR_IN_BITS +
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
