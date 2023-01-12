#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pycde import Clock, Input, Output, System
from pycde import module, generator, types

import glob
import pathlib
import shutil

__dir__ = pathlib.Path(__file__).parent

# Parameters for AXI4-Lite interface
axil_addr_width = 24
axil_data_width = 32


def XrtBSP(user_module):
  """Use the Xilinx RunTime (XRT) shell to implement ESI services and build an
  image or emulation package."""

  @module
  class top:
    ap_clk = Clock()
    ap_resetn = Input(types.i1)

    # AXI4-Lite slave interface
    s_axi_control_AWVALID = Input(types.i1)
    s_axi_control_AWREADY = Output(types.i1)
    s_axi_control_AWADDR = Input(types.int(32))
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

      rst = ~ports.ap_resetn

      # Splice in the user's code
      # NOTE: the clock is `ports.ap_clk`
      #       and reset is `ports.ap_resetn` which is active low
      user_module(clk=ports.ap_clk, rst=rst)

      # Copy additional sources
      sys: System = System.current()
      sys.add_packaging_step(top.package)

      # Tie off outputs.
      ports.s_axi_control_AWREADY = 0
      ports.s_axi_control_WREADY = 0
      ports.s_axi_control_ARREADY = 0
      ports.s_axi_control_RVALID = 0
      ports.s_axi_control_RDATA = 0
      ports.s_axi_control_RRESP = 0
      ports.s_axi_control_BVALID = 0
      ports.s_axi_control_BRESP = 0

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
      tcl_template = env.get_template("xrt_package.tcl.j2")
      dst_tcl = sys.hw_output_dir / "xrt_package.tcl"
      dst_tcl.open("w").write(tcl_template.render(system_name=sys.name))

      shutil.copy(__dir__ / "xrt.ini", sys.output_directory / "xrt.ini")
      shutil.copy(__dir__ / "xsim.tcl", sys.output_directory / "xsim.tcl")

      runtime_dir = sys.output_directory / "runtime" / sys.name
      shutil.copy(__dir__ / "xrt_api.py", runtime_dir / "xrt.py")
      shutil.copy(__dir__ / "EsiXrtPython.cpp",
                  sys.sys_runtime_output_dir / "EsiXrtPython.cpp")

  return top
