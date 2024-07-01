//===- Cosim_MMIO.sv - ESI cosim low-level MMIO module ------*- verilog -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// If a cosim design is using the low-level interface, instantiate this (ONCE)
// to get access to the MMIO commands. For convenience, they are presented in
// AXI lite signaling with 32-bit addressing and 64-bit data.
//
//===----------------------------------------------------------------------===//

import Cosim_DpiPkg::*;

module Cosim_MMIO
(
  input  logic clk,
  input  logic rst,

  // MMIO read: address channel.
  output reg          arvalid,
  input  logic        arready,
  output reg   [31:0] araddr,

  // MMIO read: data response channel.
  input  logic        rvalid,
  output reg          rready,
  input  logic [31:0] rdata,
  input  logic [1:0]  rresp,

  // MMIO write: address channel.
  output reg          awvalid,
  input  logic        awready,
  output reg   [31:0] awaddr,

  // MMIO write: data channel.
  output reg          wvalid,
  input  logic        wready,
  output reg   [31:0] wdata,

  // MMIO write: write response channel.
  input  logic        bvalid,
  output reg          bready,
  input  logic [1:0]  bresp
);

  // Registration logic -- run once.
  bit Initialized = 0;
  always@(posedge clk) begin
    if (!Initialized) begin
      automatic int rc = cosim_mmio_register();
      assert (rc == 0);
      Initialized <= 1;
    end
  end

  // MMIO read: address channel.
  always@(posedge clk) begin
    if (rst) begin
      arvalid <= 0;
    end else begin
      if (arvalid && arready)
        arvalid <= 0;
      if (!arvalid) begin
        automatic int rc = cosim_mmio_read_tryget(araddr);
        arvalid <= rc == 0;
      end
    end
  end

  // MMIO read: response channel.
  assign rready = 1;
  always@(posedge clk) begin
    if (rvalid)
      cosim_mmio_read_respond(rdata, {6'b0, rresp});
  end

  // MMIO write: address and data channels.
  always@(posedge clk) begin
    if (rst) begin
      awvalid <= 0;
      wvalid <= 0;
    end else begin
      if (awvalid && awready)
        awvalid <= 0;
      if (wvalid && wready)
        wvalid <= 0;
      if (!awvalid && !awvalid) begin
        automatic int rc = cosim_mmio_write_tryget(awaddr, wdata);
        awvalid <= rc == 0;
        wvalid  <= rc == 0;
      end
    end
  end

  // MMIO write: response (error) channel.
  assign bready = 1;
  always@(posedge clk) begin
    if (bvalid)
      cosim_mmio_write_respond({6'b0, bresp});
  end

endmodule
