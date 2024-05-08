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
from pycde import Clock, Input, Module, generator, types
from pycde.bsp import XrtBSP

import sys


class Main(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    pass


gendir = sys.argv[1]
s = pycde.System(XrtBSP(Main),
                 name="ESILoopback",
                 output_directory=gendir,
                 sw_api_langs=["python"])
s.run_passes(debug=True)
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

# TOP:         __ESI_Manifest_ROM ESI_Manifest_ROM (
# TOP:           .clk     (ap_clk),
# TOP:           .address (rom_address),
# TOP:           .data    (_ESI_Manifest_ROM_data)
# TOP:         );

# TOP:         Main Main (
# TOP:           .clk (ap_clk),
# TOP:           .rst (inv_ap_resetn)
# TOP:         );

# TOP:       endmodule
