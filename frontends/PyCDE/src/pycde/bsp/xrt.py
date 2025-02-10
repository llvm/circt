#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import Clock, Input, InputChannel, Output, OutputChannel
from ..constructs import Mux, Wire
from ..module import Module, generator
from ..signals import BitsSignal, Struct
from ..system import System
from ..support import clog2
from ..types import Array, Bits, Channel, UInt
from .. import esi

from .common import (ChannelEngineService, ChannelHostMem, ChannelMMIO,
                     DummyFromHostEngine, Reset)
from .dma import OneItemBuffersToHost

import glob
import pathlib
import shutil

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


class MMIOAxiWriteCombine(Module):
  """MMIO AXI Lite writes on XRT are 32 bits, but the MMIO service expects 64.
  Furthermore, there are two separate channels for writes: address and data. All
  four transactions must take place before an ESI MMIO write is considered
  complete."""

  clk = Clock()
  rst = Reset()
  write_address = InputChannel(Bits(AxiMMIOAddrWidth))
  write_data = InputChannel(Bits(32))

  cmd = OutputChannel(MMIOIntermediateCmd)
  write_resp = OutputChannel(Bits(2))

  @generator
  def build(ports):
    write_joined = Channel.join(ports.write_address, ports.write_data)
    write, resp = write_joined.fork(ports.clk, ports.rst)
    ports.write_resp = resp.transform(lambda x: Bits(2)(0))

    sel = Wire(Bits(1))
    [write_lo_chan, write_hi_chan] = esi.ChannelDemux(write, sel, 2)

    write_lo = esi.Mailbox(write.type)("write_lo",
                                       clk=ports.clk,
                                       rst=ports.rst,
                                       input=write_lo_chan)
    write_hi = esi.Mailbox(write.type)("write_hi",
                                       clk=ports.clk,
                                       rst=ports.rst,
                                       input=write_hi_chan)

    # The correct order of the write is low bits, high bits. Detect and mitigate
    # some runtime sync errors by checking the addresses.
    sel.assign((write_lo.valid & (write_lo.data.a[2] == Bits(1)(0))) |
               (write_hi.valid & (write_hi.data.a[2] == Bits(1)(1))))

    joined = Channel.join(write_lo.output, write_hi.output)
    cmd = joined.transform(lambda x: MMIOIntermediateCmd(
        write=Bits(1)(1),
        offset=(x.a.a & ~Bits(AxiMMIOAddrWidth)(0x7)).as_uint(32),
        data=Bits(64)(BitsSignal.concat([x.b.b, x.a.b])),
        upper=Bits(1)(0)))
    ports.cmd = cmd


class MMIOAxiReadWriteMux(Module):
  clk = Clock()
  rst = Reset()
  read_address = InputChannel(Bits(AxiMMIOAddrWidth))
  write_address = InputChannel(Bits(AxiMMIOAddrWidth))
  write_data = InputChannel(Bits(32))
  write_resp = OutputChannel(Bits(2))

  cmd = OutputChannel(esi.MMIOReadWriteCmdType)
  sel = OutputChannel(MMIOSel)

  @generator
  def build(ports):
    read_cmd = ports.read_address.transform(lambda x: MMIOIntermediateCmd(
        write=Bits(1)(0),
        offset=(x & ~Bits(AxiMMIOAddrWidth)(0x7)).as_uint(32),
        data=Bits(64)(0),
        upper=(x[0x2])))
    write_combine = MMIOAxiWriteCombine("writeCombine",
                                        clk=ports.clk,
                                        rst=ports.rst,
                                        write_address=ports.write_address,
                                        write_data=ports.write_data)
    ports.write_resp = write_combine.write_resp
    merged_cmd = read_cmd.type.merge(read_cmd, write_combine.cmd)

    merged_cmd_a, merged_cmd_b = merged_cmd.fork(ports.clk, ports.rst)
    ports.sel = merged_cmd_b.transform(
        lambda x: MMIOSel(write=x.write, upper=x.upper))
    ports.cmd = merged_cmd_a.transform(lambda x: esi.MMIOReadWriteCmdType({
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

  @generator
  def build(ports):
    data_sel = Channel.join(ports.data, ports.sel)
    data_sel_ready = Wire(Bits(1))
    data_sel_data, data_sel_valid = data_sel.unwrap(data_sel_ready)
    data = data_sel_data.a
    sel = data_sel_data.b

    # Read channel output
    read_data = AXI_Lite_Read_Resp(data=Mux(sel.upper, data[0:32], data[32:64]),
                                   resp=Bits(2)(0))
    read_valid = data_sel_valid & ~sel.write & data_sel_ready
    read_chan, read_ready = Channel(AXI_Lite_Read_Resp).wrap(
        read_data, read_valid)
    ports.read_data = read_chan

    # Write response channel output
    write_resp_data = data[0:2]
    write_resp_valid = data_sel_valid & sel.write & data_sel_ready
    write_resp_chan, write_resp_ready = Channel(Bits(2)).wrap(
        write_resp_data, write_resp_valid)
    ports.write_resp = write_resp_chan

    # Only if both are ready do we accept data.
    data_sel_ready.assign(read_ready & write_resp_ready)


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
  - Vitis spins up a number of jobs and can easily consume all available memory.
    - Specify the JOBS make variable to limit the number of jobs.
  - To adjust the desired clock frequency, set the FREQ (in MHz) make variable.
  """

  class XrtTop(Module):
    ap_clk = Clock()
    ap_resetn = Input(Bits(1))

    HostMemAddressWidth = 64
    HostMemIdWidth = 8
    HostMemDataWidth = 512

    ############################################################################
    # AXI4-Lite slave interface for MMIO
    ############################################################################

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

    ############################################################################
    # AXI4 master interface for host DMA
    ############################################################################

    # Write side
    #   Address req channel
    m_axi_gmem_AWVALID = Output(Bits(1))
    m_axi_gmem_AWREADY = Input(Bits(1))
    m_axi_gmem_AWADDR = Output(Bits(HostMemAddressWidth))
    m_axi_gmem_AWID = Output(Bits(HostMemIdWidth))
    m_axi_gmem_AWLEN = Output(Bits(8))
    m_axi_gmem_AWSIZE = Output(Bits(3))
    m_axi_gmem_AWBURST = Output(Bits(2))

    #   Data req channel
    m_axi_gmem_WVALID = Output(Bits(1))
    m_axi_gmem_WREADY = Input(Bits(1))
    m_axi_gmem_WDATA = Output(Bits(HostMemDataWidth))
    m_axi_gmem_WSTRB = Output(Bits(HostMemDataWidth // 8))
    m_axi_gmem_WLAST = Output(Bits(1))

    #   Resp channel
    m_axi_gmem_BVALID = Input(Bits(1))
    m_axi_gmem_BREADY = Output(Bits(1))
    m_axi_gmem_BRESP = Input(Bits(2))
    m_axi_gmem_BID = Input(Bits(HostMemIdWidth))

    # Read side
    #   Address req channel
    m_axi_gmem_ARVALID = Output(Bits(1))
    m_axi_gmem_ARREADY = Input(Bits(1))
    m_axi_gmem_ARADDR = Output(UInt(HostMemAddressWidth))
    m_axi_gmem_ARID = Output(UInt(HostMemIdWidth))
    m_axi_gmem_ARLEN = Output(UInt(8))
    m_axi_gmem_ARSIZE = Output(Bits(3))
    m_axi_gmem_ARBURST = Output(Bits(2))

    #   Data resp channel
    m_axi_gmem_RVALID = Input(Bits(1))
    m_axi_gmem_RREADY = Output(Bits(1))
    m_axi_gmem_RDATA = Input(Bits(HostMemDataWidth))
    m_axi_gmem_RLAST = Input(Bits(1))
    m_axi_gmem_RID = Input(UInt(HostMemIdWidth))
    m_axi_gmem_RRESP = Input(Bits(2))

    @generator
    def construct(ports):
      System.current().platform = "fpga"
      clk = ports.ap_clk
      rst = ~ports.ap_resetn

      ChannelEngineService(OneItemBuffersToHost, DummyFromHostEngine)(
          None, appid=esi.AppID("__channel_engines"), clk=clk, rst=rst)

      # Set up the MMIO service and tie it to the AXI-lite channels.
      read_address, arready = Channel(ports.s_axi_control_ARADDR.type).wrap(
          ports.s_axi_control_ARADDR, ports.s_axi_control_ARVALID)
      ports.s_axi_control_ARREADY = arready
      write_address, awready = Channel(ports.s_axi_control_AWADDR.type).wrap(
          ports.s_axi_control_AWADDR, ports.s_axi_control_AWVALID)
      ports.s_axi_control_AWREADY = awready
      write_data, wready = Channel(ports.s_axi_control_WDATA.type).wrap(
          ports.s_axi_control_WDATA, ports.s_axi_control_WVALID)
      ports.s_axi_control_WREADY = wready

      user_module(clk=clk, rst=rst)

      data = Wire(Channel(esi.MMIODataType))
      rw_mux = MMIOAxiReadWriteMux(clk=clk,
                                   rst=rst,
                                   read_address=read_address,
                                   write_address=write_address,
                                   write_data=write_data)
      sel = rw_mux.sel.buffer(clk, rst, 1)
      rw_demux = MMIOAxiReadWriteDemux(clk=clk, rst=rst, data=data, sel=sel)

      cmd, froms = esi.MMIO.read_write.type.pack(cmd=rw_mux.cmd)
      data.assign(froms["data"])

      ChannelMMIO(esi.MMIO,
                  appid=esi.AppID("__xrt_mmio"),
                  clk=clk,
                  rst=rst,
                  cmd=cmd)

      rdata, rvalid = rw_demux.read_data.unwrap(ports.s_axi_control_RREADY)
      ports.s_axi_control_RVALID = rvalid
      ports.s_axi_control_RDATA = rdata.data
      ports.s_axi_control_RRESP = rdata.resp

      wrresp_data, wrresp_valid = rw_mux.write_resp.unwrap(
          ports.s_axi_control_BREADY)
      ports.s_axi_control_BVALID = wrresp_valid
      ports.s_axi_control_BRESP = wrresp_data

      # Construct the host memory interface.
      XrtTop.construct_hostmem(ports)

      # Copy additional sources
      sys = System.current()
      sys.add_packaging_step(XrtTop.package)

    @staticmethod
    def construct_hostmem(ports):
      """Construct the host memory interface"""
      rst = ~ports.ap_resetn

      # Instantiate a hostmem service generator which multiplexes requests to a
      # single read and single write channel.
      HostMemDataWidthBytes = XrtTop.HostMemDataWidth // 8
      hostmem = ChannelHostMem(read_width=XrtTop.HostMemDataWidth,
                               write_width=XrtTop.HostMemDataWidth)(
                                   decl=esi.HostMem,
                                   appid=esi.AppID("__xrt_hostmem"),
                                   clk=ports.ap_clk,
                                   rst=rst)

      ##########################################################################
      # Read path
      ##########################################################################

      read_resp_data_type = hostmem.read.type.resp.inner
      read_resp_wire = Wire(Channel(read_resp_data_type))
      read_req = hostmem.read.unpack(resp=read_resp_wire)['req']

      # Unwrap the read request channel and tie it to the AXI4 master interface.
      read_req_data, read_req_valid = read_req.unwrap(ports.m_axi_gmem_ARREADY)
      ports.m_axi_gmem_ARVALID = read_req_valid
      ports.m_axi_gmem_ARADDR = read_req_data.address
      ports.m_axi_gmem_ARID = read_req_data.tag
      # TODO: Test reads > HostMemDataWidthBytes. I'm pretty sure this is wrong
      # for lengths >512bits but initially we won't be transferring >512 bits.
      ports.m_axi_gmem_ARSIZE = clog2(HostMemDataWidthBytes)
      ports.m_axi_gmem_ARLEN = (read_req_data.length /
                                HostMemDataWidthBytes).as_uint(8)
      ports.m_axi_gmem_ARBURST = Bits(2)(1)  # INCR

      # Wrap up the read response data into a channel.
      read_resp_data = read_resp_data_type({
          "tag": ports.m_axi_gmem_RID,
          "data": ports.m_axi_gmem_RDATA,
      })
      read_resp, read_resp_ready = read_resp_wire.type.wrap(
          read_resp_data, ports.m_axi_gmem_RVALID)
      ports.m_axi_gmem_RREADY = read_resp_ready
      read_resp_wire.assign(read_resp)

      ##########################################################################
      # Write path
      ##########################################################################

      write_ack_wire = Wire(Channel(esi.HostMem.TagType))
      write_req = hostmem.write.unpack(ackTag=write_ack_wire)['req']
      address_req, data_req = write_req.fork(ports.ap_clk, rst)

      # Write address channels
      address_req_data, address_req_valid = address_req.unwrap(
          ports.m_axi_gmem_AWREADY)
      ports.m_axi_gmem_AWVALID = address_req_valid
      ports.m_axi_gmem_AWADDR = address_req_data.address.as_bits()
      ports.m_axi_gmem_AWID = address_req_data.tag.as_bits()
      ports.m_axi_gmem_AWLEN = Bits(8)(0)  # Single xact burst.
      ports.m_axi_gmem_AWSIZE = clog2(HostMemDataWidthBytes)
      ports.m_axi_gmem_AWBURST = Bits(2)(1)  # INCR

      # Write data channels
      data_req_data, data_req_valid = data_req.unwrap(ports.m_axi_gmem_WREADY)
      ports.m_axi_gmem_WVALID = data_req_valid
      ports.m_axi_gmem_WDATA = data_req_data.data

      wstrb_lookup_table_len = HostMemDataWidthBytes
      wstrb_lookup_table = Array(Bits(HostMemDataWidthBytes),
                                 wstrb_lookup_table_len)([
                                     Bits(HostMemDataWidthBytes)(2**vb - 1)
                                     for vb in range(wstrb_lookup_table_len)
                                 ])
      # TODO: Single writes only support HostMemDataWidthBytes bytes. Fine for
      # now since the gearbox takes data down to 512 bits. The `valid_bytes`
      # field, however, is 8 bits, so it theoretically supports up to 256 bytes.
      # We should probably break this up into multiple transactions.
      ports.m_axi_gmem_WSTRB = wstrb_lookup_table[
          data_req_data.valid_bytes.as_uint(clog2(wstrb_lookup_table_len))]
      ports.m_axi_gmem_WLAST = Bits(1)(1)

      # Write ack channel
      write_ack, write_ack_ready = Channel(esi.HostMem.TagType).wrap(
          ports.m_axi_gmem_BID.as_uint(), ports.m_axi_gmem_BVALID)
      write_ack_wire.assign(write_ack)
      ports.m_axi_gmem_BREADY = write_ack_ready

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
      shutil.copy(__dir__ / "xrt_vitis.cfg",
                  sys.output_directory / "xrt_vitis.cfg")
      shutil.copy(__dir__ / "xsim.tcl", sys.output_directory / "xsim.tcl")

  return XrtTop
