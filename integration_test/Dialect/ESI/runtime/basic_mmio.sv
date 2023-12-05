// REQUIRES: esi-cosim, esi-runtime, rtl-sim
// RUN: %python esi-cosim-runner.py --exec %s.py %s

// Test the low level cosim MMIO functionality. This test has 1024 64-bit
// registers as a memory. It is an error to write to register 0.

import Cosim_DpiPkg::*;

module top(
    input logic clk,
    input logic rst
);

  // MMIO read: address channel.
  logic        arvalid;
  logic        arready;
  logic [31:0] araddr;

  // MMIO read: data response channel.
  reg          rvalid;
  logic        rready;
  reg   [31:0] rdata;
  reg   [1:0]  rresp;

  // MMIO write: address channel.
  logic        awvalid;
  reg          awready;
  logic [31:0] awaddr;

  // MMIO write: data channel.
  logic        wvalid;
  reg          wready;
  logic [31:0] wdata;

  // MMIO write: write response channel.
  reg          bvalid;
  logic        bready;
  reg   [1:0]  bresp;

  Cosim_MMIO mmio (
    .clk(clk),
    .rst(rst),
    .arvalid(arvalid),
    .arready(arready),
    .araddr(araddr),
    .rvalid(rvalid),
    .rready(rready),
    .rdata(rdata),
    .rresp(rresp),
    .awvalid(awvalid),
    .awready(awready),
    .awaddr(awaddr),
    .wvalid(wvalid),
    .wready(wready),
    .wdata(wdata),
    .bvalid(bvalid),
    .bready(bready),
    .bresp(bresp)
  );

  reg [31:0] regs [1023:0];

  assign arready = 1;
  assign rdata = regs[araddr >> 3];
  assign rresp = araddr == 0 ? 3 : 0;
  always@(posedge clk) begin
    if (rst) begin
      rvalid <= 0;
    end else begin
      if (arvalid)
        rvalid <= 1;
      if (rready && rvalid)
        rvalid <= 0;
    end
  end

  wire write = awvalid && wvalid && !bvalid;
  assign awready = write;
  assign wready = write;
  always@(posedge clk) begin
    if (rst) begin
      bvalid <= 0;
    end else begin
      if (bvalid && bready)
        bvalid <= 0;
      if (write) begin
        bvalid <= 1;
        bresp <= awaddr == 0 ? 3 : 0;
        regs[awaddr >> 3] <= wdata;
      end
    end
  end

endmodule
