// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Copyright (c) Microsoft Corporation. All rights reserved.
// =============================================================================
// Package: CosimCore_EndpointBasePkg
//
// Authors:
// - John Demme (john.demme@microsoft.com)
//
// Based on code written by:
// - Andrew Lenharth (andrew.lenharth@microsoft.com)
//
// Description:
//   Main cosim <--> dpi bridge module
// =============================================================================

import Cosim_DpiPkg::*;

module Cosim_Endpoint
#(
   parameter int ENDPOINT_ID,
   parameter longint ESI_TYPE_ID,
   parameter int TYPE_SIZE_BITS
)
(
   input  logic clk,
   input  logic rstn,

   output logic DataOutValid,
   input  logic DataOutReady,
   output logic[TYPE_SIZE_BITS-1:0] DataOut,

   input  logic DataInValid,
   output logic DataInReady,
   input  logic [TYPE_SIZE_BITS-1:0] DataIn
);

   localparam int TYPE_SIZE_BYTES = int'((TYPE_SIZE_BITS+7)/8);
   localparam int TYPE_SIZE_BITS_DIFF = TYPE_SIZE_BITS % 8; // The number of bits over a byte
   localparam int TYPE_SIZE_BYTES_FLOOR = int'(TYPE_SIZE_BITS/8);
   localparam int TYPE_SIZE_BYTES_FLOOR_IN_BITS = TYPE_SIZE_BYTES_FLOOR * 8;
   bit Initialized;

   // Handle initialization logic
   always@(posedge clk)
   begin
      // We've been instructed to start AND we're uninitialized
      if (!Initialized)
      begin
         int rc;
         rc = cosim_init();
         if (rc != 0)
            $error("Cosim init failed (%d)", rc);
         rc = cosim_ep_register(ENDPOINT_ID, ESI_TYPE_ID, TYPE_SIZE_BYTES);
         if (rc != 0)
            $error("Cosim endpoint (%d) register failed: %d", ENDPOINT_ID, rc);
         Initialized = 1'b1;
      end
   end

   /// *******************
   /// Data out management
   ///
   byte unsigned DataOutBuffer[TYPE_SIZE_BYTES-1:0];
   always@(posedge clk)
   begin
      if (rstn && Initialized)
      begin
         if (DataOutValid && DataOutReady) // A transfer occurred
         begin
            DataOutValid <= 1'b0;
         end

         if (!DataOutValid || DataOutReady)
         begin
            int data_limit;
            int rc;

            data_limit = TYPE_SIZE_BYTES;
            rc = cosim_ep_tryget(ENDPOINT_ID, DataOutBuffer, data_limit);
            if (rc < 0)
            begin
               $error("cosim_ep_tryget(%d, *, %d -> %d) returned an error (%d)",
                  ENDPOINT_ID, TYPE_SIZE_BYTES, data_limit, rc);
            end
            else if (rc > 0)
            begin
               $error("cosim_ep_tryget(%d, *, %d -> %d) had data left over! (%d)",
                  ENDPOINT_ID, TYPE_SIZE_BYTES, data_limit, rc);
            end
            else if (rc == 0)
            begin
               if (data_limit == TYPE_SIZE_BYTES)
               begin
                  DataOutValid <= 1'b1;
               end
               else if (data_limit == 0)
               begin
                  // No message
               end
               else
               begin
                  $error("cosim_ep_tryget(%d, *, %d -> %d) did not load entire buffer!",
                     ENDPOINT_ID, TYPE_SIZE_BYTES, data_limit);
               end
            end
         end
      end
      else
      begin
         DataOutValid <= 1'b0;
      end
   end

   // Assign packed output bit array from unpacked byte array
   genvar iOut;
   generate
      for (iOut=0; iOut<TYPE_SIZE_BYTES_FLOOR; iOut++)
      begin
         assign DataOut[((iOut+1)*8)-1:iOut*8] = DataOutBuffer[iOut];
      end
      if (TYPE_SIZE_BITS_DIFF != 0)
         assign DataOut[TYPE_SIZE_BYTES_FLOOR_IN_BITS + TYPE_SIZE_BITS_DIFF - 1 : TYPE_SIZE_BYTES_FLOOR_IN_BITS]
            = DataOutBuffer[TYPE_SIZE_BYTES-1][TYPE_SIZE_BITS_DIFF-1:0];
   endgenerate

   initial
   begin
   $display("TYPE_SIZE_BITS: %d", TYPE_SIZE_BITS);
   $display("TYPE_SIZE_BYTES: %d", TYPE_SIZE_BYTES);
   $display("TYPE_SIZE_BITS_DIFF: %d", TYPE_SIZE_BITS_DIFF);
   $display("TYPE_SIZE_BYTES_FLOOR: %d", TYPE_SIZE_BYTES_FLOOR);
   $display("TYPE_SIZE_BYTES_FLOOR_IN_BITS: %d", TYPE_SIZE_BYTES_FLOOR_IN_BITS);
   end


   /// **********************
   /// Data in management
   ///
   assign DataInReady = 1'b1;
   byte unsigned DataInBuffer[TYPE_SIZE_BYTES-1:0];

   always@(posedge clk)
   begin
      if (rstn && Initialized)
      begin
         if (DataInValid)
         begin
            int rc;
            rc = cosim_ep_tryput(ENDPOINT_ID, DataInBuffer, TYPE_SIZE_BYTES);
            if (rc != 0)
            begin
               $error("cosim_ep_tryput(%d, *, %d) = %d Error! (Data lost)",
                  ENDPOINT_ID, TYPE_SIZE_BYTES, rc);
            end
         end
      end
   end

   // Assign packed input bit array to unpacked byte array
   genvar iIn;
   generate
      for (iIn=0; iIn<TYPE_SIZE_BYTES_FLOOR; iIn++)
      begin
         assign DataInBuffer[iIn] = DataIn[((iIn+1)*8)-1:iIn*8];
      end
      if (TYPE_SIZE_BITS_DIFF != 0)
         assign DataInBuffer[TYPE_SIZE_BYTES-1][TYPE_SIZE_BITS_DIFF-1:0] =
            DataIn[TYPE_SIZE_BYTES_FLOOR_IN_BITS + TYPE_SIZE_BITS_DIFF - 1 : TYPE_SIZE_BYTES_FLOOR_IN_BITS];
   endgenerate

endmodule