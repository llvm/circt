#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pycde import Clock, Input, Output, System
from pycde import module, generator, esi, types
from ..value import BitVectorValue
from ..constructs import Reg, Wire
from ..module import _GeneratorPortAccess
from .fifo import SimpleXilinxFifo

import glob
from io import FileIO
import json
import math
import pathlib
import shutil
from typing import List, Tuple

__dir__ = pathlib.Path(__file__).parent


# Signals from master
def axil_in_type(addr_width, data_width):
  return types.struct({
      "awvalid": types.i1,
      "awaddr": types.int(addr_width),
      "wvalid": types.i1,
      "wdata": types.int(data_width),
      "wstrb": types.int(data_width // 8),
      "arvalid": types.i1,
      "araddr": types.int(addr_width),
      "rready": types.i1,
      "bready": types.i1
  })


# Signals to master
def axil_out_type(data_width):
  return types.struct({
      "awready": types.i1,
      "wready": types.i1,
      "arready": types.i1,
      "rvalid": types.i1,
      "rdata": types.int(data_width),
      "rresp": types.i2,
      "bvalid": types.i1,
      "bresp": types.i2
  })


def generate_mmio_map(channels, base_offset=0x0) -> Tuple[List, List]:
  curr_offset = base_offset

  def get_reg(req, write):
    nonlocal curr_offset
    width = req.type.inner_type.bitwidth

    reg = {
        "size": width,
        "offset": curr_offset,
        "name": "_".join(req.client_name),
        "client_path": req.client_name,
        "type": "direct_mmio"
    }
    curr_offset += 32 * math.ceil(width / 32)
    # A write reg requires an extra byte for control.
    if write:
      curr_offset += 8
    return reg

  from_host_regs = [get_reg(req, False) for req in channels.to_client_reqs]
  to_host_regs = [get_reg(req, True) for req in channels.to_server_reqs]
  return (from_host_regs, to_host_regs)


def output_tcl(from_host_regs, to_host_regs, os: FileIO):
  from jinja2 import Environment, FileSystemLoader, StrictUndefined

  env = Environment(loader=FileSystemLoader(str(__dir__)),
                    undefined=StrictUndefined)
  template = env.get_template("xrt_package.tcl.j2")
  os.write(
      template.render(from_host_regs=from_host_regs, to_host_regs=to_host_regs))


def XrtBSP(user_module):
  # Parameters for AXI4-Lite interface
  axil_addr_width = 24
  axil_data_width = 32

  @esi.ServiceImplementation(None)
  class XrtService:
    clk = Clock(types.i1)
    rst = Input(types.i1)

    axil_in = Input(axil_in_type(axil_addr_width, axil_data_width))
    axil_out = Output(axil_out_type(axil_data_width))

    @generator
    def generate(ports: _GeneratorPortAccess, channels):
      clk = ports.clk
      rst = ports.rst

      from_host_regs, to_host_regs = generate_mmio_map(channels)
      sys: System = System.current()
      output_tcl(from_host_regs, to_host_regs,
                 (sys.hw_output_dir / "esi_bsp.tcl").open("w"))
      chan_md_file = (sys.sys_runtime_output_dir /
                      "xrt_mmio_descriptor.json").open("w")
      chan_md_file.write(
          json.dumps(
              {
                  "from_host_regs": from_host_regs,
                  "to_host_regs": to_host_regs
              },
              indent=2))

      write_ready = Wire(types.i1)
      address_reg = ports.axil_in.awaddr.reg(clk=clk,
                                             rst=rst,
                                             ce=(write_ready &
                                                 ports.axil_in.awvalid))

      from_host_fifos_full = []
      for req, mmio_spec in zip(channels.to_client_reqs, from_host_regs):
        outch_ready_wire = Wire(types.i1)
        fifo = SimpleXilinxFifo(req.type.inner, depth=32,
                                almost_full=4)(clk=clk,
                                               rst=rst,
                                               rd_en=outch_ready_wire,
                                               wr_en=0,
                                               wr_data=0)
        outch, outch_ready = req.type.wrap(fifo.rd_data, ~fifo.empty)
        outch_ready_wire.assign(outch_ready)
        req.assign(outch)

        from_host_fifos_full.append(fifo.almost_full.reg(cycles=2))

      from_host_full = BitVectorValue.or_reduce(from_host_fifos_full)
      write_ready.assign(~from_host_full)

      ports.axil_out = axil_out_type(axil_data_width)({
          "awready": write_ready,
          "wready": write_ready,
          "arready": 1,
          "rvalid": 1,
          "rdata": 0,
          "rresp": 0,
          "bvalid": ports.axil_in.wvalid,
          "bresp": 0
      })

  @module
  class top:
    ap_clk = Clock()
    ap_resetn = Input(types.i1)

    # AXI4-Lite slave interface
    s_axi_control_AWVALID = Input(types.i1)
    s_axi_control_AWREADY = Output(types.i1)
    s_axi_control_AWADDR = Input(types.int(axil_addr_width))
    s_axi_control_WVALID = Input(types.i1)
    s_axi_control_WREADY = Output(types.i1)
    s_axi_control_WDATA = Input(types.int(axil_data_width))
    s_axi_control_WSTRB = Input(types.int(axil_data_width // 8))
    s_axi_control_ARVALID = Input(types.i1)
    s_axi_control_ARREADY = Output(types.i1)
    s_axi_control_ARADDR = Input(types.int(axil_addr_width))
    s_axi_control_RVALID = Output(types.i1)
    s_axi_control_RREADY = Input(types.i1)
    s_axi_control_RDATA = Output(types.int(axil_data_width))
    s_axi_control_RRESP = Output(types.i2)
    s_axi_control_BVALID = Output(types.i1)
    s_axi_control_BREADY = Input(types.i1)
    s_axi_control_BRESP = Output(types.i2)

    @generator
    def construct(ports):

      axil_in_sig = axil_in_type(axil_addr_width, axil_data_width)({
          "awvalid": ports.s_axi_control_AWVALID,
          "awaddr": ports.s_axi_control_AWADDR,
          "wvalid": ports.s_axi_control_WVALID,
          "wdata": ports.s_axi_control_WDATA,
          "wstrb": ports.s_axi_control_WSTRB,
          "arvalid": ports.s_axi_control_ARVALID,
          "araddr": ports.s_axi_control_ARADDR,
          "rready": ports.s_axi_control_RREADY,
          "bready": ports.s_axi_control_BREADY,
      })

      rst = ~ports.ap_resetn

      xrt = XrtService(clk=ports.ap_clk, rst=rst, axil_in=axil_in_sig)

      axil_out = xrt.axil_out

      # AXI-Lite control
      ports.s_axi_control_AWREADY = axil_out['awready']
      ports.s_axi_control_WREADY = axil_out['wready']
      ports.s_axi_control_ARREADY = axil_out['arready']
      ports.s_axi_control_RVALID = axil_out['rvalid']
      ports.s_axi_control_RDATA = axil_out['rdata']
      ports.s_axi_control_RRESP = axil_out['rresp']
      ports.s_axi_control_BVALID = axil_out['bvalid']
      ports.s_axi_control_BRESP = axil_out['bresp']

      # Splice in the user's code
      # NOTE: the clock is `ports.ap_clk`
      #       and reset is `ports.ap_resetn` which is active low
      user_module(clk=ports.ap_clk, rst=rst)

      # Copy additional sources
      sys: System = System.current()
      sys.add_packaging_step(top.package)

    @staticmethod
    def package(sys: System):
      from jinja2 import Environment, FileSystemLoader, StrictUndefined

      sv_sources = glob.glob(str(__dir__ / '*.sv'))
      tcl_sources = glob.glob(str(__dir__ / '*.tcl'))
      for source in sv_sources + tcl_sources:
        shutil.copy(source, sys.hw_output_dir)

      env = Environment(loader=FileSystemLoader(str(__dir__)),
                        undefined=StrictUndefined)
      template = env.get_template("Makefile.xrt.j2")
      dst_makefile = sys.output_directory / "Makefile.xrt"
      dst_makefile.open("w").write(template.render(system_name=sys.name))

      runtime_dir = sys.output_directory / "runtime" / sys.name
      so_sources = glob.glob(str(__dir__ / '*.so'))
      for so in so_sources:
        shutil.copy(so, runtime_dir)
      shutil.copy(__dir__ / "xrt_api.py", runtime_dir / "xrt.py")
      shutil.copy(__dir__ / "EsiXrtPython.cpp",
                  sys.sys_runtime_output_dir / "EsiXrtPython.cpp")

  return top
