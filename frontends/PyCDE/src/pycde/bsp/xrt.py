#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ast import Str
from os import write
from ..common import Clock, Input, InputChannel, Output, OutputChannel
from ..constructs import ControlReg, Mux, NamedWire, Reg, Wire
from ..handshake import Func, demux, cmerge
from ..module import Module, generator
from ..signals import BitsSignal, BundleSignal, Struct
from ..system import System
from ..types import Array, Bits, Channel, StructType, UInt
from .. import esi

from .common import ChannelMMIO, Reset

import glob
import pathlib
import shutil

from typing import Dict, Tuple

__dir__ = pathlib.Path(__file__).parent


class AXI_Lite_Read_Resp(Struct):
  data: Bits(32)
  resp: Bits(2)


AxiMMIOAddrWidth = 20


class MMIOSel(Struct):
  write: Bits(1)
  upper: Bits(1)


class MMIOIntermediateCmd(Struct):
  upper: Bits(1)
  write: Bits(1)
  offset: UInt(32)
  data: esi.MMIODataType


class MMIOAxiReadWriteMux(Module):
  clk = Clock()
  rst = Reset()
  read_address = InputChannel(Bits(AxiMMIOAddrWidth))
  write_address = InputChannel(Bits(AxiMMIOAddrWidth))
  write_data = InputChannel(Bits(32))

  cmd = OutputChannel(esi.MMIOReadWriteCmdType)
  sel = OutputChannel(MMIOSel)

  @generator
  def build(ports):
    read_cmd = ports.read_address.transform(lambda x: MMIOIntermediateCmd(
        write=Bits(1)(0),
        offset=(x & ~Bits(AxiMMIOAddrWidth)(0x7)).as_uint(32),
        data=Bits(64)(0),
        upper=(x[0x2])))
    read_cmd_a, read_cmd_b = read_cmd.fork(ports.clk, ports.rst)
    ports.sel = read_cmd_b.transform(
        lambda x: MMIOSel(write=x.write, upper=x.upper))
    ports.cmd = read_cmd_a.transform(lambda x: esi.MMIOReadWriteCmdType({
        "write": x.write,
        "offset": x.offset,
        "data": x.data,
    }))


class MMIOAxiReadWriteDemux(Module):
  clk = Clock()
  rst = Reset()
  sel = InputChannel(MMIOSel)
  data = InputChannel(esi.MMIODataType)

  read_data = OutputChannel(AXI_Lite_Read_Resp)
  # write_resp = OutputChannel(Bits(2))

  @generator
  def build(ports):
    data_sel = Channel.join(ports.data, ports.sel)
    data_sel_ready = Wire(Bits(1))
    data_sel_data, data_sel_valid = data_sel.unwrap(data_sel_ready)
    data = data_sel_data.a
    sel = data_sel_data.b
    read_data = AXI_Lite_Read_Resp(data=Mux(sel.upper, data[0:32], data[32:64]),
                                   resp=Bits(2)(0))
    read_valid = data_sel_valid & data_sel_ready
    read_chan, read_ready = Channel(AXI_Lite_Read_Resp).wrap(
        read_data, read_valid)
    data_sel_ready.assign(read_ready)
    ports.read_data = read_chan


# class AxiMMIO(Func):
#   # Upstream command channels.
#   cmd = Output(esi.MMIOReadWriteCmdType)
#   data = Input(esi.MMIODataType)

#   read_address = Input(Bits(20))  # MMIO read: address channel.
#   read_data = Output(AXI_Lite_Read_Resp)  # MMIO read: data response channel.

#   write_address = Input(Bits(20))  # MMIO write: address channel.
#   write_data = Input(Bits(32))  # MMIO write: data channel.
#   write_resp = Output(Bits(2))  # MMIO write: write response channel.

#   @generator
#   def build(ports):
#     araddr = ports.read_address
#     cmd_addr = (araddr & ~Bits(20)(0x7))
#     read_cmd = esi.MMIOReadWriteCmdType({
#         "write": Bits(1)(0),
#         "offset": cmd_addr.as_uint(32),
#         "data": Bits(64)(0)
#     })

#     awaddr = ports.write_address
#     write_sel = awaddr[0x2]
#     write_sel.name = "write_sel"
#     write_data_lo, write_data_hi = demux(write_sel, ports.write_data)
#     write_data = BitsSignal.concat([write_data_lo, write_data_hi])
#     write_cmd = esi.MMIOReadWriteCmdType({
#         "write": Bits(1)(1),
#         "offset": (awaddr & ~Bits(20)(0x7)).as_uint(32),
#         "data": write_data
#     })
#     write_cmd = NamedWire(write_cmd, "write_cmd")

#     cmd, out_idx = cmerge(write_cmd, read_cmd)
#     cmd = NamedWire(cmd, "cmd")
#     ports.cmd = cmd
#     out_idx = NamedWire(out_idx, "out_idx")
#     write_resp, read_data = demux(out_idx, ports.data)
#     write_resp = NamedWire(write_resp, "write_resp")
#     read_data = NamedWire(read_data, "read_data")

#     resp_data_lo = read_data[0:32]
#     resp_data_hi = read_data[32:64]
#     read_req_higher_bits = araddr[0x2]
#     resp_data = Mux(read_req_higher_bits, resp_data_lo, resp_data_hi)
#     ports.read_data = AXI_Lite_Read_Resp({
#         "data": resp_data,
#         "resp": Bits(2)(0)
#     })

#     ports.write_resp = write_resp[0:2]

# class AxiMMIO(Module):
#   """MMIO service implementation with an AXI-lite protocol. This assumes a 20
#   bit address bus for 1MB of addressable MMIO space. Which should be fine for
#   now, though nothing should assume this limit. It also only supports 32-bit
#   aligned accesses and just throws away the lower two bits of address.

#   Only allows for one outstanding request at a time. If a client doesn't return
#   a response, the MMIO service will hang. TODO: add some kind of timeout.

#   Implementation-defined MMIO layout:
#     - 0x0: 0 constant
#     - 0x4: 0 constant
#     - 0x8: Magic number low (0xE5100E51)
#     - 0xC: Magic number high (random constant: 0x207D98E5)
#     - 0x10: ESI version number (0)
#     - 0x14: Location of the manifest ROM (absolute address)

#     - 0x100: Start of MMIO space for requests. Mapping is contained in the
#              manifest so can be dynamically queried.

#     - addr(Manifest ROM) + 0: Size of compressed manifest
#     - addr(Manifest ROM) + 4: Start of compressed manifest

#   This layout _should_ be pretty standard, but different BSPs may have various
#   different restrictions.
#   """

#   clk = Clock()
#   rst = Input(Bits(1))

#   # Translate MMIO AXI Lite into a standardized bundle MMIO bundle interface.
#   # This output is the translated MMIO bundle to hand off to the ChannelMMIO
#   # service.
#   cmd_output = Output(esi.MMIO.read_write.type)

#   read_address = InputChannel(Bits(20))  # MMIO read: address channel.
#   read_data = OutputChannel(
#       AXI_Lite_Read_Resp)  # MMIO read: data response channel.
#   # address_write = InputChannel(Bits(20))  # MMIO write: address channel.
#   # data_write = InputChannel(Bits(32))  # MMIO write: data channel.
#   # write_resp = OutputChannel(Bits(2))  # MMIO write: write response channel.

#   @generator
#   def generate(ports):
#     data = Wire(Channel(esi.MMIODataType))
#     cntrl = AxiController(clk=ports.clk,
#                           rst=ports.rst,
#                           data=data,
#                           read_address=ports.read_address)
#     ports.read_data = cntrl.read_data
#     cmd, froms = esi.MMIO.read_write.type.pack(cmd=cntrl.cmd)
#     data.assign(froms["data"])
#     ports.cmd_output = cmd

# @generator
# def generate(ports):
#   # Construct the output bundle.
#   cmd_data = Wire(esi.MMIOReadWriteCmdType)
#   cmd_valid = Wire(Bits(1))
#   cmd_ch, cmd_ready = Channel(cmd_data.type).wrap(cmd_data, cmd_valid)
#   cmd_bundle, froms = esi.MMIO.read_write.type.pack(cmd=cmd_ch)
#   resp_ch = froms["data"]
#   ports.cmd_output = cmd_bundle

#   resp_ready = Wire(Bits(1))
#   resp_data, resp_valid = resp_ch.unwrap(resp_ready)
#   resp_data_lo = resp_data.as_bits()[0:32]
#   resp_data_hi = resp_data.as_bits()[32:64]

#   arready = Wire(Bits(1))
#   araddr, arvalid = ports.read_address.unwrap(arready)

#   #   rd_address_xact = Wire(Bits(1))
#   #   wr_cmd_xact = Wire(Bits(1))
#   #   # Is the current transaction a read or write?
#   #   axi_read = ports.arvalid
#   #   axi_last_read = axi_read.reg(ports.clk,
#   #                                ports.rst,
#   #                                name="axi_last_read",
#   #                                ce=rd_address_xact | wr_cmd_xact)

#   #   rd_address_xact.assign(ports.arvalid & cmd_ready & axi_read)
#   #   rd_address_reg = ports.araddr.reg(ports.clk, ports.rst, ce=rd_address_xact)

#   #   # Wire up the AXI read response port.
#   #   resp_ready.assign(ports.rready)
#   #   ports.rdata = Mux(rd_address_reg[0x3], resp_data_hi, resp_data_lo)
#   #   ports.rvalid = resp_valid & axi_last_read

#   # Wire up the AXI read address port.
#   cmd_data.assign(
#       esi.MMIOReadWriteCmdType({
#           "write": Bits(1)(0),
#           "offset": (araddr & Bits(20)(0x7)).as_uint(32),
#           "data": Bits(64)(0),
#       }))

# #   cmd_valid.assign(ports.arvalid)
# #   ports.arready = cmd_ready
# #   ports.rresp = Bits(2)(0)

# #   # Wire up the AXI write side. Since AXI lite data is only 32 bits (in
# #   # contrast to ESI's 64 bit data), an ESI write is split into two AXI writes.
# #   # Below that, an AXI write is actually two separate transactions -- one for
# #   # the address and one for the data.

# #   wr_cmd_xact.assign(0)

# #   wr_address_hi = ports

# #   ports.awready = Bits(1)(0)
# #   ports.wready = Bits(1)(0)
# #   ports.bvalid = Bits(1)(0)
# #   ports.bresp = Bits(2)(0)


def XrtChannelTop(user_module):

  class XrtChannelTop(Module):
    clk = Clock()
    rst = Input(Bits(1))

    read_address = InputChannel(Bits(20))  # MMIO read: address channel.
    read_data = OutputChannel(
        AXI_Lite_Read_Resp)  # MMIO read: data response channel.

    write_address = InputChannel(Bits(20))  # MMIO write: address channel.
    write_data = InputChannel(Bits(32))  # MMIO write: data channel.
    # write_resp = OutputChannel(Bits(2))  # MMIO write: write response channel.

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)

      data = Wire(Channel(esi.MMIODataType))
      rw_mux = MMIOAxiReadWriteMux(clk=ports.clk,
                                   rst=ports.rst,
                                   read_address=ports.read_address,
                                   write_address=ports.write_address,
                                   write_data=ports.write_data)
      sel = rw_mux.sel.buffer(ports.clk, ports.rst, 1)
      rw_demux = MMIOAxiReadWriteDemux(clk=ports.clk,
                                       rst=ports.rst,
                                       data=data,
                                       sel=sel)
      ports.read_data = rw_demux.read_data
      # ports.write_resp = rw_demux.write_resp

      cmd, froms = esi.MMIO.read_write.type.pack(cmd=rw_mux.cmd)
      data.assign(froms["data"])

      ChannelMMIO(esi.MMIO,
                  appid=esi.AppID("__xrt_mmio"),
                  clk=ports.clk,
                  rst=ports.rst,
                  cmd=cmd)

  return XrtChannelTop


def XrtBSP(user_module):
  """Use the Xilinx RunTime (XRT) shell to implement ESI services and build an
  image or emulation package.
  How to use this BSP:
  - Wrap your top PyCDE module in `XrtBSP`.
  - Run your script. This BSP will write a 'build package' to the output dir.
  This package contains a Makefile.xrt.mk which (given a proper Vitis dev
  environment) will compile a hw image or hw_emu image. It is a free-standing
  build package -- you do not need PyCDE installed on the same machine as you
  want to do the image build.
  - To build the `hw` image, run 'make -f Makefile.xrt TARGET=hw'. If you want
  an image which runs on an Azure NP-series instance, run the 'azure' target
  (requires an Azure subscription set up with as per
  https://learn.microsoft.com/en-us/azure/virtual-machines/field-programmable-gate-arrays-attestation).
  This target requires a few environment variables to be set (which the Makefile
  will tell you about).
  - To build a hw emulation image, run make with TARGET=hw_emu.
    - At "runtime" set XCL_EMULATION_MODE=hw_emu.
  - Validated ONLY on Vitis 2023.1. Known to NOT work with Vitis <2022.1.
  """

  class XrtTop(Module):
    ap_clk = Clock()
    ap_resetn = Input(Bits(1))

    # AXI4-Lite slave interface
    s_axi_control_AWVALID = Input(Bits(1))
    s_axi_control_AWREADY = Output(Bits(1))
    s_axi_control_AWADDR = Input(Bits(20))
    s_axi_control_WVALID = Input(Bits(1))
    s_axi_control_WREADY = Output(Bits(1))
    s_axi_control_WDATA = Input(Bits(32))
    s_axi_control_WSTRB = Input(Bits(32 // 8))
    s_axi_control_ARVALID = Input(Bits(1))
    s_axi_control_ARREADY = Output(Bits(1))
    s_axi_control_ARADDR = Input(Bits(20))
    s_axi_control_RVALID = Output(Bits(1))
    s_axi_control_RREADY = Input(Bits(1))
    s_axi_control_RDATA = Output(Bits(32))
    s_axi_control_RRESP = Output(Bits(2))
    s_axi_control_BVALID = Output(Bits(1))
    s_axi_control_BREADY = Input(Bits(1))
    s_axi_control_BRESP = Output(Bits(2))

    @generator
    def construct(ports):
      System.current().platform = "fpga"

      rst = ~ports.ap_resetn

      XrtChannelsTmplInst = XrtChannelTop(user_module)

      read_address, arready = Channel(
          XrtChannelsTmplInst.read_address.type).wrap(
              ports.s_axi_control_ARADDR, ports.s_axi_control_ARVALID)
      ports.s_axi_control_ARREADY = arready
      write_address, awready = Channel(
          XrtChannelsTmplInst.write_address.type).wrap(
              ports.s_axi_control_AWADDR, ports.s_axi_control_AWVALID)
      ports.s_axi_control_AWREADY = awready
      write_data, wready = Channel(XrtChannelsTmplInst.write_data.type).wrap(
          ports.s_axi_control_WDATA, ports.s_axi_control_WVALID)
      ports.s_axi_control_WREADY = wready

      xrt_channels = XrtChannelsTmplInst(
          clk=ports.ap_clk,
          rst=rst,
          read_address=read_address,
          write_address=write_address,
          write_data=write_data,
      )

      rdata, rvalid = xrt_channels.read_data.unwrap(ports.s_axi_control_RREADY)
      ports.s_axi_control_RVALID = rvalid
      ports.s_axi_control_RDATA = rdata.data
      ports.s_axi_control_RRESP = rdata.resp

      ports.s_axi_control_BVALID = Bits(1)(0)
      ports.s_axi_control_BRESP = Bits(2)(0)

      # Copy additional sources
      sys: System = System.current()
      sys.add_packaging_step(XrtTop.package)

    @staticmethod
    def package(sys: System):
      """Assemble a 'build' package which includes all the necessary build
      collateral (about which we are aware), build/debug scripts, and the
      generated runtime."""

      sv_sources = glob.glob(str(__dir__ / '*.sv'))
      tcl_sources = glob.glob(str(__dir__ / '*.tcl'))
      for source in sv_sources + tcl_sources:
        shutil.copy(source, sys.hw_output_dir)

      shutil.copy(__dir__ / "Makefile.xrt.mk",
                  sys.output_directory / "Makefile.xrt.mk")
      shutil.copy(__dir__ / "xrt_package.tcl",
                  sys.output_directory / "xrt_package.tcl")
      shutil.copy(__dir__ / "xrt.ini", sys.output_directory / "xrt.ini")
      shutil.copy(__dir__ / "xsim.tcl", sys.output_directory / "xsim.tcl")

  return XrtTop


def XrtCosimBSP(user_module):
  """Use the XRT BSP with AXI channels implemented with ESI cosim. Mostly useful
  for debugging the actual Xrt BSP."""

  class XrtCosimBSP(Module):
    clk = Clock()
    rst = Input(Bits(1))

    @generator
    def build(ports):
      XrtChannelsTmplInst = XrtChannelTop(user_module)
      read_address = esi.ChannelService.from_host(esi.AppID("mmio_axi_rdaddr"),
                                                  UInt(32))
      read_address = read_address.transform(lambda x: x.as_bits(20))

      write_address = esi.ChannelService.from_host(esi.AppID("mmio_axi_wraddr"),
                                                   UInt(32))
      write_address = write_address.transform(lambda x: x.as_bits(20))
      write_data = esi.ChannelService.from_host(esi.AppID("mmio_axi_wrdata"),
                                                UInt(32))
      write_data = write_data.transform(lambda x: x.as_bits(32))
      xrt = XrtChannelsTmplInst(clk=ports.clk,
                                rst=ports.rst,
                                read_address=read_address,
                                write_address=write_address,
                                write_data=write_data)
      esi.ChannelService.to_host(
          esi.AppID("mmio_axi_rddata"),
          xrt.read_data.transform(lambda x: x.data.as_uint()))
      # esi.ChannelService.to_host(
      #     esi.AppID("mmio_axi_wrresp"),
      #     xrt.write_resp.transform(lambda x: x.as_uint(8)))

  from .cosim import CosimBSP
  return CosimBSP(XrtCosimBSP)
