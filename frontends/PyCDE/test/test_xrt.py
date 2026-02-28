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

# TOP:  Main Main (
# TOP:  MMIOAxiReadWriteMux MMIOAxiReadWriteMux (
# TOP:  MMIOAxiReadWriteDemux MMIOAxiReadWriteDemux (
# TOP:  HeaderMMIO_manifest_loc{{.+}} HeaderMMIO (
# TOP:  ESI_Manifest_ROM_Wrapper ESI_Manifest_ROM_Wrapper (
# TOP:   HostmemReadProcessorImpl HostmemReadProcessorImpl (
# TOP:   HostMemWriteProcessorImpl HostMemWriteProcessorImpl (
