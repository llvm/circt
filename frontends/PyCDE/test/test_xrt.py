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
from pycde import (Clock, Input, InputChannel, Module, OutputChannel, generator,
                   types)
from pycde import esi
from pycde.bsp import XrtBSP
from pycde.constructs import Wire
from pycde.dialects import comb

import sys


@esi.ServiceDecl
class HostComms:
  to_host = esi.ToServer(types.any)
  from_host = esi.FromServer(types.any)
  req_resp = esi.ToFromServer(to_server_type=types.i16,
                              to_client_type=types.i32)


class Producer(Module):
  clk = Input(types.i1)
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = HostComms.from_host("loopback_in", types.i32)
    ports.int_out = chan


class Consumer(Module):
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    HostComms.to_host(ports.int_in, "loopback_out")


class LoopbackInOutAdd7(Module):

  @generator
  def construct(ports):
    loopback = Wire(types.channel(types.i16))
    from_host = HostComms.req_resp(loopback, "loopback_inout")
    ready = Wire(types.i1)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    # TODO: clean this up with PyCDE overloads (they're currently a little bit
    # broken for this use-case).
    data = comb.AddOp(data, types.i16(7))
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class Mid(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)

    LoopbackInOutAdd7()


class Top(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    Mid(clk=ports.clk, rst=ports.rst)


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
