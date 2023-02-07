# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: ls %t/hw/top.sv
# RUN: ls %t/hw/Top.sv
# RUN: ls %t/hw/services.json
# RUN: ls %t/hw/ESILoopback.tcl
# RUN: ls %t/hw/filelist.f
# RUN: ls %t/hw/xsim.tcl
# RUN: ls %t/hw/xrt_package.tcl
# RUN: ls %t/runtime/ESILoopback/common.py
# RUN: ls %t/runtime/ESILoopback/__init__.py
# RUN: ls %t/runtime/ESILoopback/xrt.py
# RUN: ls %t/Makefile.xrt
# RUN: ls %t/xrt.ini
# RUN: ls %t/xsim.tcl
# RUN: ls %t/ESILoopback
# RUN: ls %t/ESILoopback/EsiXrtPython.cpp

# RUN: FileCheck %s --input-file %t/hw/top.sv --check-prefix=TOP

import pycde
from pycde import Clock, Input, Module, generator, types
from pycde.bsp import XrtBSP

import sys


class Top(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    pass


gendir = sys.argv[1]
s = pycde.System(XrtBSP(Top),
                 name="ESILoopback",
                 output_directory=gendir,
                 sw_api_langs=["python"])
s.run_passes(debug=True)
s.compile()
s.package()

# TOP-LABEL: module top
# TOP:         #(parameter __INST_HIER = "INSTANTIATE_WITH_INSTANCE_PATH") (
# TOP:         input         ap_clk,
# TOP:                       ap_resetn,
# TOP:                       s_axi_control_AWVALID,
# TOP:         input  [31:0] s_axi_control_AWADDR,
# TOP:         input         s_axi_control_WVALID,
# TOP:         input  [31:0] s_axi_control_WDATA,
# TOP:         input  [3:0]  s_axi_control_WSTRB,
# TOP:         input         s_axi_control_ARVALID,
# TOP:         input  [23:0] s_axi_control_ARADDR,
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
# TOP:         Top #(
# TOP:           .__INST_HIER({__INST_HIER, ".Top"})
# TOP:         ) Top (
# TOP:           .clk (ap_clk),
# TOP:           .rst (~ap_resetn)
# TOP:         );
# TOP:         assign s_axi_control_AWREADY = 1'h0;
# TOP:         assign s_axi_control_WREADY = 1'h0;
# TOP:         assign s_axi_control_ARREADY = 1'h0;
# TOP:         assign s_axi_control_RVALID = 1'h0;
# TOP:         assign s_axi_control_RDATA = 32'h0;
# TOP:         assign s_axi_control_RRESP = 2'h0;
# TOP:         assign s_axi_control_BVALID = 1'h0;
# TOP:         assign s_axi_control_BRESP = 2'h0;
# TOP:       endmodule
