#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import write
from typing import Dict, Tuple, Type

from ..signals import BundleSignal
from ..common import AppID, Clock, Input, Output
from ..module import Module, generator
from ..system import System
from ..types import (Bits, Bundle, BundledChannel, Channel, ChannelDirection,
                     StructType, UInt)
from ..constructs import ControlReg, NamedWire, Reg, Wire
from .. import esi

from .common import ChannelHostMem, ChannelMMIO

from ..circt import ir
from ..circt.dialects import esi as raw_esi

from pathlib import Path

__root_dir__ = Path(__file__).parent.parent


def CosimBSP(user_module: Type[Module]) -> Module:
  """Wrap and return a cosimulation 'board support package' containing
  'user_module'"""

  class ESI_Cosim_UserTopWrapper(Module):
    """Wrap the user module along with 'standard' service generators so that
    those generators can issue their own service requests to be picked up by the
    actual top-level catch-all cosim service generator."""

    clk = Clock()
    rst = Input(Bits(1))

    # If this gets changed, update 'Cosim.cpp' in the runtime.
    HostMemWidth = 64

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)

      mmio_read_write = esi.FuncService.get_coerced(
          esi.AppID("__cosim_mmio_read_write"), esi.MMIO.read_write.type)
      ChannelMMIO(esi.MMIO,
                  appid=esi.AppID("__cosim_mmio"),
                  clk=ports.clk,
                  rst=ports.rst,
                  cmd=mmio_read_write)

      # Instantiate a hostmem service generator which multiplexes requests to a
      # either a single read or write channel. Get those channels and transform
      # them into callbacks.
      hostmem = ChannelHostMem(
          read_width=ESI_Cosim_UserTopWrapper.HostMemWidth,
          write_width=ESI_Cosim_UserTopWrapper.HostMemWidth)(
              decl=esi.HostMem,
              appid=esi.AppID("__cosim_hostmem"),
              clk=ports.clk,
              rst=ports.rst)
      resp_wire = Wire(
          Channel(
              StructType([
                  ("tag", UInt(8)),
                  ("data", Bits(ESI_Cosim_UserTopWrapper.HostMemWidth)),
              ])))
      req = hostmem.read.unpack(resp=resp_wire)['req']
      data = esi.CallService.call(esi.AppID("__cosim_hostmem_read"), req,
                                  resp_wire.type)
      resp_wire.assign(data)

      ack_wire = Wire(Channel(UInt(8)))
      write_req = hostmem.write.unpack(ackTag=ack_wire)['req']
      ack_tag = esi.CallService.call(esi.AppID("__cosim_hostmem_write"),
                                     write_req, UInt(8))
      ack_wire.assign(ack_tag)

  class ESI_Cosim_Top(Module):
    clk = Clock()
    rst = Input(Bits(1))

    @generator
    def build(ports):
      System.current().platform = "cosim"
      ESI_Cosim_UserTopWrapper(clk=ports.clk, rst=ports.rst)

      raw_esi.ServiceInstanceOp(result=[],
                                appID=AppID("cosim")._appid,
                                service_symbol=None,
                                impl_type=ir.StringAttr.get("cosim"),
                                inputs=[ports.clk.value, ports.rst.value])

  return ESI_Cosim_Top
