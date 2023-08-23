// REQUIRES: esi-cosim
// RUN: esi-cosim-runner.py %s %s
// P Y: import IPython
// P Y: IPython.embed()
// PY: import esi_cosim
// PY: c = esi_cosim.LowLevel(rpcschemapath, simhostport)
// PY: r = c.low.readMMIO(40).wait()
// PY: print(f"data resp: 0x{r.data:x}")
// PY: r = c.low.readMMIO(0).wait()
// PY: print(f"data resp: 0x{r.data:x}")

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
  reg   [63:0] rdata;
  reg   [1:0]  rresp;

  // MMIO write: address channel.
  logic        awvalid;
  logic        awready;
  logic [31:0] awaddr;

  // MMIO write: data channel.
  logic        wvalid;
  logic        wready;
  logic [63:0] wdata;

  // MMIO write: write response channel.
  logic        bvalid;
  logic        bready;
  logic [31:0] bdata;

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
    .bdata(bdata)
  );

  assign arready = 1;
  assign rdata = {araddr, 32'b0};
  assign rresp = araddr == 0 ? 1 : 0;
  always@(posedge clk) begin
    if (arvalid)
      rvalid <= 1;
    if (rready && rvalid)
      rvalid <= 0;
  end

endmodule
