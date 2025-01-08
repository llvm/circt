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
# TOP:         input         ap_clk,
# TOP:                       ap_resetn,
# TOP:                       s_axi_control_AWVALID,
# TOP:         input  [19:0] s_axi_control_AWADDR,
# TOP:         input         s_axi_control_WVALID,
# TOP:         input  [31:0] s_axi_control_WDATA,
# TOP:         input  [3:0]  s_axi_control_WSTRB,
# TOP:         input         s_axi_control_ARVALID,
# TOP:         input  [19:0] s_axi_control_ARADDR,
# TOP:         input         s_axi_control_RREADY,
# TOP:                       s_axi_control_BREADY,
# TOP:         output        s_axi_control_AWREADY,
# TOP:                       s_axi_control_WREADY,
# TOP:                       s_axi_control_ARREADY,
# TOP:                       s_axi_control_RVALID,
# TOP:         output [31:0] s_axi_control_RDATA,
# TOP:         output [1:0]  s_axi_control_RRESP,
# TOP:         output        s_axi_control_BVALID,
# TOP:         output [1:0]  s_axi_control_BRESP

# TOP:    wire AXI_Lite_Read_Resp _XrtChannelTop_read_data;
# TOP:    XrtChannelTop XrtChannelTop (
# TOP:      .clk                 (ap_clk),
# TOP:      .rst                 (~ap_resetn),
# TOP:      .read_address        (s_axi_control_ARADDR),
# TOP:      .read_address_valid  (s_axi_control_ARVALID),
# TOP:      .write_address       (s_axi_control_AWADDR),
# TOP:      .write_address_valid (s_axi_control_AWVALID),
# TOP:      .write_data          (s_axi_control_WDATA),
# TOP:      .write_data_valid    (s_axi_control_WVALID),
# TOP:      .read_data_ready     (s_axi_control_RREADY),
# TOP:      .write_resp_ready    (s_axi_control_BREADY),
# TOP:      .read_address_ready  (s_axi_control_ARREADY),
# TOP:      .write_address_ready (s_axi_control_AWREADY),
# TOP:      .write_data_ready    (s_axi_control_WREADY),
# TOP:      .read_data           (_XrtChannelTop_read_data),
# TOP:      .read_data_valid     (s_axi_control_RVALID),
# TOP:      .write_resp          (s_axi_control_BRESP),
# TOP:      .write_resp_valid    (s_axi_control_BVALID)
# TOP:    );
# TOP:    assign s_axi_control_RDATA = _XrtChannelTop_read_data.data;
# TOP:    assign s_axi_control_RRESP = _XrtChannelTop_read_data.resp;

# TOP:       endmodule

# CHTOP:    typedef struct packed {logic [31:0] data; logic [1:0] resp; } AXI_Lite_Read_Resp;
# CHTOP:    typedef struct packed {logic write; logic upper; } MMIOSel;
# CHTOP:    typedef
# CHTOP:      struct packed {logic upper; logic write; logic [31:0] offset; logic [63:0] data; }
# CHTOP:      MMIOIntermediateCmd;

# CHTOP:  module XrtChannelTop(
# CHTOP:    input                     clk,
# CHTOP:                              rst,
# CHTOP:    input  [19:0]             read_address,
# CHTOP:    input                     read_address_valid,
# CHTOP:    input  [19:0]             write_address,
# CHTOP:    input                     write_address_valid,
# CHTOP:    input  [31:0]             write_data,
# CHTOP:    input                     write_data_valid,
# CHTOP:                              read_data_ready,
# CHTOP:                              write_resp_ready,
# CHTOP:    output                    read_address_ready,
# CHTOP:                              write_address_ready,
# CHTOP:                              write_data_ready,
# CHTOP:    output AXI_Lite_Read_Resp read_data,
# CHTOP:    output                    read_data_valid,
# CHTOP:    output [1:0]              write_resp,
# CHTOP:    output                    write_resp_valid
# CHTOP:  );

# CHTOP:  Main Main (
# CHTOP:  MMIOAxiReadWriteMux MMIOAxiReadWriteMux (
# CHTOP:  MMIOAxiReadWriteDemux MMIOAxiReadWriteDemux (
# CHTOP:  HeaderMMIO_manifest_loc65536 HeaderMMIO (
# CHTOP:  ESI_Manifest_ROM_Wrapper ESI_Manifest_ROM_Wrapper (
