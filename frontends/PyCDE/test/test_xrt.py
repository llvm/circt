# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: ls %t/hw/XrtTop.sv
# RUN: ls %t/hw/Main.sv
# RUN: ls %t/hw/ESILoopback.tcl
# RUN: ls %t/hw/filelist.f
# RUN: ls %t/hw/xsim.tcl
# RUN: ls %t/hw/xrt_package.tcl
# RUN: ls %t/Makefile.xrt.mk
# RUN: ls %t/xrt.ini
# RUN: ls %t/xsim.tcl

# RUN: FileCheck %s --input-file %t/hw/XrtTop.sv --check-prefix=TOP
# RUN: FileCheck %s --input-file %t/hw/XrtChannelTop.sv --check-prefix=CHTOP

import pycde
from pycde import Clock, Input, Module, generator
from pycde.types import Bits
from pycde.bsp import XrtBSP

import sys


class Main(Module):
  clk = Clock()
  rst = Input(Bits(1))

  @generator
  def construct(ports):
    pass


gendir = sys.argv[1]
s = pycde.System(XrtBSP(Main), name="ESILoopback", output_directory=gendir)
s.compile()
s.package()

# TOP-LABEL: module XrtTop(
# TOP-NEXT:    input         ap_clk,
# TOP-NEXT:                  ap_resetn,
# TOP-NEXT:                   s_axi_control_AWVALID,
# TOP-NEXT:    input  [19:0]  s_axi_control_AWADDR,
# TOP-NEXT:    input          s_axi_control_WVALID,
# TOP-NEXT:    input  [31:0]  s_axi_control_WDATA,
# TOP-NEXT:    input  [3:0]   s_axi_control_WSTRB,
# TOP-NEXT:    input          s_axi_control_ARVALID,
# TOP-NEXT:    input  [19:0]  s_axi_control_ARADDR,
# TOP-NEXT:    input          s_axi_control_RREADY,
# TOP-NEXT:                   s_axi_control_BREADY,
# TOP-NEXT:                   m_axi_gmem_AWREADY,
# TOP-NEXT:                   m_axi_gmem_WREADY,
# TOP-NEXT:                   m_axi_gmem_BVALID,
# TOP-NEXT:    input  [1:0]   m_axi_gmem_BRESP,
# TOP-NEXT:    input  [7:0]   m_axi_gmem_BID,
# TOP-NEXT:    input          m_axi_gmem_ARREADY,
# TOP-NEXT:                   m_axi_gmem_RVALID,
# TOP-NEXT:    input  [511:0] m_axi_gmem_RDATA,
# TOP-NEXT:    input          m_axi_gmem_RLAST,
# TOP-NEXT:    input  [7:0]   m_axi_gmem_RID,
# TOP-NEXT:    input  [1:0]   m_axi_gmem_RRESP,
# TOP-NEXT:    output         s_axi_control_AWREADY,
# TOP-NEXT:                   s_axi_control_WREADY,
# TOP-NEXT:                   s_axi_control_ARREADY,
# TOP-NEXT:                   s_axi_control_RVALID,
# TOP-NEXT:    output [31:0]  s_axi_control_RDATA,
# TOP-NEXT:    output [1:0]   s_axi_control_RRESP,
# TOP-NEXT:    output         s_axi_control_BVALID,
# TOP-NEXT:    output [1:0]   s_axi_control_BRESP,
# TOP-NEXT:    output         m_axi_gmem_AWVALID,
# TOP-NEXT:    output [63:0]  m_axi_gmem_AWADDR,
# TOP-NEXT:    output [7:0]   m_axi_gmem_AWID,
# TOP-NEXT:                   m_axi_gmem_AWLEN,
# TOP-NEXT:    output [2:0]   m_axi_gmem_AWSIZE,
# TOP-NEXT:    output [1:0]   m_axi_gmem_AWBURST,
# TOP-NEXT:    output         m_axi_gmem_WVALID,
# TOP-NEXT:    output [511:0] m_axi_gmem_WDATA,
# TOP-NEXT:    output [63:0]  m_axi_gmem_WSTRB,
# TOP-NEXT:    output         m_axi_gmem_WLAST,
# TOP-NEXT:                   m_axi_gmem_BREADY,
# TOP-NEXT:                   m_axi_gmem_ARVALID,
# TOP-NEXT:    output [63:0]  m_axi_gmem_ARADDR,
# TOP-NEXT:    output [7:0]   m_axi_gmem_ARID,
# TOP-NEXT:                   m_axi_gmem_ARLEN,
# TOP-NEXT:    output [2:0]   m_axi_gmem_ARSIZE,
# TOP-NEXT:    output [1:0]   m_axi_gmem_ARBURST,
# TOP-NEXT:    output         m_axi_gmem_RREADY

# TOP:    wire AXI_Lite_Read_Resp _XrtChannelTop_mmio_read_data;
# TOP:    XrtChannelTop XrtChannelTop (
# TOP:      .clk                 (ap_clk),
# TOP:      .rst                 (~ap_resetn),
# TOP:      .mmio_read_address        (s_axi_control_ARADDR),
# TOP:      .mmio_read_address_valid  (s_axi_control_ARVALID),
# TOP:      .mmio_write_address       (s_axi_control_AWADDR),
# TOP:      .mmio_write_address_valid (s_axi_control_AWVALID),
# TOP:      .mmio_write_data          (s_axi_control_WDATA),
# TOP:      .mmio_write_data_valid    (s_axi_control_WVALID),
# TOP:      .mmio_read_data_ready     (s_axi_control_RREADY),
# TOP:      .mmio_write_resp_ready    (s_axi_control_BREADY),
# TOP:      .mmio_read_address_ready  (s_axi_control_ARREADY),
# TOP:      .mmio_write_address_ready (s_axi_control_AWREADY),
# TOP:      .mmio_write_data_ready    (s_axi_control_WREADY),
# TOP:      .mmio_read_data           (_XrtChannelTop_mmio_read_data),
# TOP:      .mmio_read_data_valid     (s_axi_control_RVALID),
# TOP:      .mmio_write_resp          (s_axi_control_BRESP),
# TOP:      .mmio_write_resp_valid    (s_axi_control_BVALID)
# TOP:    );
# TOP:   HostmemReadProcessorImpl HostmemReadProcessorImpl (
# TOP:   HostMemWriteProcessorImpl HostMemWriteProcessorImpl (

# TOP:   assign s_axi_control_RDATA = _XrtChannelTop_mmio_read_data.data;
# TOP:   assign s_axi_control_RRESP = _XrtChannelTop_mmio_read_data.resp;
# TOP:   assign m_axi_gmem_ARADDR = _HostmemReadProcessorImpl_upstream_req.address;
# TOP:   assign m_axi_gmem_ARID = _HostmemReadProcessorImpl_upstream_req.tag;
# TOP: endmodule

# CHTOP:    typedef struct packed {logic [31:0] data; logic [1:0] resp; } AXI_Lite_Read_Resp;
# CHTOP:    typedef struct packed {logic write; logic upper; } MMIOSel;
# CHTOP:    typedef
# CHTOP:      struct packed {logic upper; logic write; logic [31:0] offset; logic [63:0] data; }
# CHTOP:      MMIOIntermediateCmd;

# CHTOP:  module XrtChannelTop(
# CHTOP:    input                     clk,
# CHTOP:                              rst,
# CHTOP:    input  [19:0]             mmio_read_address,
# CHTOP:    input                     mmio_read_address_valid,
# CHTOP:    input  [19:0]             mmio_write_address,
# CHTOP:    input                     mmio_write_address_valid,
# CHTOP:    input  [31:0]             mmio_write_data,
# CHTOP:    input                     mmio_write_data_valid,
# CHTOP:                              mmio_read_data_ready,
# CHTOP:                              mmio_write_resp_ready,
# CHTOP:    output                    mmio_read_address_ready,
# CHTOP:                              mmio_write_address_ready,
# CHTOP:                              mmio_write_data_ready,
# CHTOP:    output AXI_Lite_Read_Resp mmio_read_data,
# CHTOP:    output                    mmio_read_data_valid,
# CHTOP:    output [1:0]              mmio_write_resp,
# CHTOP:    output                    mmio_write_resp_valid
# CHTOP:  );

# CHTOP:  Main Main (
# CHTOP:  MMIOAxiReadWriteMux MMIOAxiReadWriteMux (
# CHTOP:  MMIOAxiReadWriteDemux MMIOAxiReadWriteDemux (
# CHTOP:  HeaderMMIO_manifest_loc65536 HeaderMMIO (
# CHTOP:  ESI_Manifest_ROM_Wrapper ESI_Manifest_ROM_Wrapper (
