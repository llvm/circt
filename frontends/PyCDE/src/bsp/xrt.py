#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import Clock, Input, Output
from ..constructs import ControlReg, Wire
from ..module import Module, generator
from ..system import System
from ..types import UInt, bit, types, Bits
from .. import esi

from .common import AxiMMIO

import glob
from io import FileIO
import pathlib
import shutil

__dir__ = pathlib.Path(__file__).parent


def XrtBSP(user_module):
  """Use the Xilinx RunTime (XRT) shell to implement ESI services and build an
  image or emulation package.
  How to use this BSP:
  - Wrap your top PyCDE module in `XrtBSP`.
  - Run your script. This BSP will write a 'build package' to the output dir.
  This package contains a Makefile.xrt which (given a proper Vitis dev
  environment) will compile a hw image or hw_emu image. It is a free-standing
  build package -- you do not need PyCDE installed on the same machine as you
  want to do the image build.
  - To build the `hw` image, run 'make -f Makefile.xrt TARGET=hw'. If you want
  an image which runs on an Azure NP-series instance, run the 'azure' target
  (requires an Azure subscription set up with as per
  https://learn.microsoft.com/en-us/azure/virtual-machines/field-programmable-gate-arrays-attestation).
  This target requires a few environment variables to be set (which the Makefile
  will tell you about).
  - To build a hw emulation image, run with TARGET=hw_emu.
  - The makefile also builds a Python plugin. To specify the python version to
  build against (if different from the version ran by 'python3' in your
  environment), set the PYTHON variable (e.g. 'PYTHON=python3.9').
  """

  XrtMaxAddr = 2**24
  id_width = 4
  axi_width = 64
  addr_width = 64

  class top(Module):
    ap_clk = Clock()
    ap_resetn = Input(Bits(1))

    # AXI4-Lite slave interface
    s_axi_control_AWVALID = Input(Bits(1))
    s_axi_control_AWREADY = Output(Bits(1))
    s_axi_control_AWADDR = Input(Bits(32))
    s_axi_control_WVALID = Input(Bits(1))
    s_axi_control_WREADY = Output(Bits(1))
    s_axi_control_WDATA = Input(Bits(32))
    s_axi_control_WSTRB = Input(Bits(32 // 8))
    s_axi_control_ARVALID = Input(Bits(1))
    s_axi_control_ARREADY = Output(Bits(1))
    s_axi_control_ARADDR = Input(Bits(32))
    s_axi_control_RVALID = Output(Bits(1))
    s_axi_control_RREADY = Input(Bits(1))
    s_axi_control_RDATA = Output(Bits(32))
    s_axi_control_RRESP = Output(Bits(2))
    s_axi_control_BVALID = Output(Bits(1))
    s_axi_control_BREADY = Input(Bits(1))
    s_axi_control_BRESP = Output(Bits(2))

    # AXI4 master interface
    m_axi_gmem_AWVALID = Output(types.i1)
    m_axi_gmem_AWREADY = Input(types.i1)
    m_axi_gmem_AWADDR = Output(types.int(32))
    m_axi_gmem_AWID = Output(types.int(id_width))
    m_axi_gmem_AWLEN = Output(types.i8)
    m_axi_gmem_AWSIZE = Output(types.i3)
    m_axi_gmem_AWBURST = Output(types.i2)
    m_axi_gmem_AWLOCK = Output(types.i2)
    m_axi_gmem_AWCACHE = Output(types.i4)
    m_axi_gmem_AWPROT = Output(types.i3)
    m_axi_gmem_AWQOS = Output(types.i4)
    m_axi_gmem_AWREGION = Output(types.i4)
    m_axi_gmem_WVALID = Output(types.i1)
    m_axi_gmem_WREADY = Input(types.i1)
    m_axi_gmem_WDATA = Output(types.int(axi_width))
    m_axi_gmem_WSTRB = Output(types.int(axi_width // 8))
    m_axi_gmem_WLAST = Output(types.i1)
    m_axi_gmem_ARVALID = Output(types.i1)
    m_axi_gmem_ARREADY = Input(types.i1)
    m_axi_gmem_ARADDR = Output(types.int(addr_width))
    m_axi_gmem_ARID = Output(types.int(id_width))
    m_axi_gmem_ARLEN = Output(types.i8)
    m_axi_gmem_ARSIZE = Output(types.i3)
    m_axi_gmem_ARBURST = Output(types.i2)
    m_axi_gmem_ARLOCK = Output(types.i2)
    m_axi_gmem_ARCACHE = Output(types.i4)
    m_axi_gmem_ARPROT = Output(types.i3)
    m_axi_gmem_ARQOS = Output(types.i4)
    m_axi_gmem_ARREGION = Output(types.i4)
    m_axi_gmem_RVALID = Input(types.i1)
    m_axi_gmem_RREADY = Output(types.i1)
    m_axi_gmem_RDATA = Input(types.int(axi_width))
    m_axi_gmem_RLAST = Input(types.i1)
    m_axi_gmem_RID = Input(types.int(id_width))
    m_axi_gmem_RRESP = Input(types.i2)
    m_axi_gmem_BVALID = Input(types.i1)
    m_axi_gmem_BREADY = Output(types.i1)
    m_axi_gmem_BRESP = Input(types.i2)
    m_axi_gmem_BID = Input(types.int(id_width))

    @generator
    def construct(ports):
      System.current().platform = "xrt"

      rst = ~ports.ap_resetn

      addr_mask = Bits(32)(XrtMaxAddr - 1)
      masked_awaddr = ports.s_axi_control_AWADDR & addr_mask
      masked_araddr = ports.s_axi_control_ARADDR & addr_mask
      xrt = AxiMMIO(
          esi.MMIO,
          appid=esi.AppID("xrt_mmio"),
          clk=ports.ap_clk,
          rst=rst,
          awvalid=ports.s_axi_control_AWVALID,
          awaddr=masked_awaddr.as_uint(),
          wvalid=ports.s_axi_control_WVALID,
          wdata=ports.s_axi_control_WDATA,
          wstrb=ports.s_axi_control_WSTRB,
          arvalid=ports.s_axi_control_ARVALID,
          araddr=masked_araddr.as_uint(),
          rready=ports.s_axi_control_RREADY,
          bready=ports.s_axi_control_BREADY,
      )

      # AXI-Lite control
      ports.s_axi_control_AWREADY = xrt.awready
      ports.s_axi_control_WREADY = xrt.wready
      ports.s_axi_control_ARREADY = xrt.arready
      ports.s_axi_control_RVALID = ports.s_axi_control_ARVALID.reg()
      ports.s_axi_control_RDATA = ports.s_axi_control_ARADDR.reg()
      ports.s_axi_control_RRESP = xrt.rresp
      ports.s_axi_control_BVALID = xrt.bvalid
      ports.s_axi_control_BRESP = xrt.bresp

      # Splice in the user's code
      # NOTE: the clock is `ports.ap_clk`
      #       and reset is `ports.ap_resetn` which is active low
      user_module(clk=ports.ap_clk, rst=rst)

      ports.m_axi_gmem_AWVALID = 0
      ports.m_axi_gmem_AWADDR = 0
      ports.m_axi_gmem_AWID = 0
      ports.m_axi_gmem_AWLEN = 0
      ports.m_axi_gmem_AWSIZE = 0
      ports.m_axi_gmem_AWBURST = 0
      ports.m_axi_gmem_AWLOCK = 0
      ports.m_axi_gmem_AWCACHE = 0
      ports.m_axi_gmem_AWPROT = 0
      ports.m_axi_gmem_AWQOS = 0
      ports.m_axi_gmem_AWREGION = 0
      ports.m_axi_gmem_WVALID = 0
      ports.m_axi_gmem_WDATA = 0
      ports.m_axi_gmem_WSTRB = 0
      ports.m_axi_gmem_WLAST = 0
      ports.m_axi_gmem_ARVALID = 0
      ports.m_axi_gmem_ARADDR = 0
      ports.m_axi_gmem_ARID = 0
      ports.m_axi_gmem_ARLEN = 0
      ports.m_axi_gmem_ARSIZE = 0
      ports.m_axi_gmem_ARBURST = 0
      ports.m_axi_gmem_ARLOCK = 0
      ports.m_axi_gmem_ARCACHE = 0
      ports.m_axi_gmem_ARPROT = 0
      ports.m_axi_gmem_ARQOS = 0
      ports.m_axi_gmem_ARREGION = 0
      ports.m_axi_gmem_RREADY = 0
      ports.m_axi_gmem_BREADY = 0

      # Copy additional sources
      sys: System = System.current()
      sys.add_packaging_step(esi.package)
      sys.add_packaging_step(top.package)

    @staticmethod
    def package(sys: System):
      """Assemble a 'build' package which includes all the necessary build
      collateral (about which we are aware), build/debug scripts, and the
      generated runtime."""

      from jinja2 import Environment, FileSystemLoader, StrictUndefined

      sv_sources = glob.glob(str(__dir__ / '*.sv'))
      tcl_sources = glob.glob(str(__dir__ / '*.tcl'))
      for source in sv_sources + tcl_sources:
        shutil.copy(source, sys.hw_output_dir)

      env = Environment(loader=FileSystemLoader(str(__dir__)),
                        undefined=StrictUndefined)
      makefile_template = env.get_template("Makefile.xrt.j2")
      dst_makefile = sys.output_directory / "Makefile.xrt"
      dst_makefile.open("w").write(
          makefile_template.render(system_name=sys.name))
      template = env.get_template("xrt_package.tcl.j2")
      dst_package_tcl = sys.hw_output_dir / "xrt_package.tcl"
      dst_package_tcl.open("w").write(
          template.render(system_name=sys.name, max_mmio_size=XrtMaxAddr))

      shutil.copy(__dir__ / "xrt.ini", sys.output_directory / "xrt.ini")
      shutil.copy(__dir__ / "xsim.tcl", sys.output_directory / "xsim.tcl")

  return top
