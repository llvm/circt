#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Tuple, Type

from ..signals import BundleSignal
from ..common import AppID, Clock, Input, Output
from ..module import Module, generator
from ..system import System
from ..types import (Bits, Bundle, BundledChannel, Channel, ChannelDirection,
                     StructType, UInt)
from ..constructs import ControlReg, NamedWire, Reg, Wire
from .. import esi

from .common import (ChannelEngineService, ChannelHostMem, ChannelMMIO)
from .dma import OneItemBuffersFromHost, OneItemBuffersToHost

from ..circt import ir
from ..circt.dialects import esi as raw_esi

from pathlib import Path

__root_dir__ = Path(__file__).parent.parent


def CosimBSP(user_module: Type[Module], emulate_dma: bool = False) -> Module:
  """Wrap and return a cosimulation 'board support package' containing
  'user_module'"""

  class ESI_Cosim_UserTopWrapper(Module):
    """Wrap the user module along with 'standard' service generators so that
    those generators can issue their own service requests to be picked up by the
    actual top-level catch-all cosim service generator."""

    clk = Clock()
    rst = Input(Bits(1))

    mmio = Input(esi.MMIO.read_write.type)

    # If this gets changed, update 'Cosim.cpp' in the runtime.
    HostMemWidth = 64

    ChannelHostMemModule = ChannelHostMem(read_width=HostMemWidth,
                                          write_width=HostMemWidth)

    hostmem_read = ChannelHostMemModule.read
    hostmem_write = ChannelHostMemModule.write

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)
      if emulate_dma:
        ChannelEngineService(OneItemBuffersToHost, OneItemBuffersFromHost)(
            None,
            appid=esi.AppID("__channel_engines"),
            clk=ports.clk,
            rst=ports.rst)

      ChannelMMIO(esi.MMIO,
                  appid=esi.AppID("__cosim_mmio"),
                  clk=ports.clk,
                  rst=ports.rst,
                  cmd=ports.mmio)

      # Instantiate a hostmem service generator which multiplexes requests to a
      # either a single read or write channel. Get those channels and transform
      # them into callbacks.
      hostmem = ESI_Cosim_UserTopWrapper.ChannelHostMemModule(
          decl=esi.HostMem,
          appid=esi.AppID("__cosim_hostmem"),
          clk=ports.clk,
          rst=ports.rst)
      ports.hostmem_read = hostmem.read
      ports.hostmem_write = hostmem.write

  class ESI_Cosim_Top(Module):
    clk = Clock()
    rst = Input(Bits(1))

    @generator
    def build(ports):
      System.current().platform = "cosim"

      mmio_read_write = esi.FuncService.get_coerced(
          esi.AppID("__cosim_mmio_read_write"), esi.MMIO.read_write.type)
      wrapper = ESI_Cosim_UserTopWrapper(clk=ports.clk,
                                         rst=ports.rst,
                                         mmio=mmio_read_write)

      resp_channel = esi.ChannelService.from_host(
          esi.AppID("__cosim_hostmem_read_resp"),
          StructType([
              ("tag", UInt(8)),
              ("data", Bits(ESI_Cosim_UserTopWrapper.HostMemWidth)),
          ]))
      req = wrapper.hostmem_read.unpack(resp=resp_channel)['req']
      esi.ChannelService.to_host(esi.AppID("__cosim_hostmem_read_req"), req)

      ack_wire = Wire(Channel(UInt(8)))
      write_req = wrapper.hostmem_write.unpack(ackTag=ack_wire)['req']
      ack_tag = esi.CallService.call(esi.AppID("__cosim_hostmem_write"),
                                     write_req, UInt(8))
      ack_wire.assign(ack_tag)

      raw_esi.ServiceInstanceOp(result=[],
                                appID=AppID("cosim")._appid,
                                service_symbol=None,
                                impl_type=ir.StringAttr.get("cosim"),
                                inputs=[ports.clk.value, ports.rst.value])

  return ESI_Cosim_Top


def CosimBSP_DMA(user_module: Type[Module]) -> Module:
  return CosimBSP(user_module, emulate_dma=True)
