#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Tuple

from ..signals import BundleSignal
from ..common import AppID, Clock, Input, Output
from ..module import Module, generator
from ..system import System
from ..types import Bits, Bundle, BundledChannel, ChannelDirection
from ..constructs import ControlReg, NamedWire, Reg, Wire
from .. import esi

from .common import ChannelMMIO

from ..circt import ir
from ..circt.dialects import esi as raw_esi

from pathlib import Path

__root_dir__ = Path(__file__).parent.parent


def CosimBSP(user_module: Module) -> Module:
  """Wrap and return a cosimulation 'board support package' containing
  'user_module'"""

  class ESI_Cosim_UserTopWrapper(Module):
    """Wrap the user module along with 'standard' service generators so that
    those generators can issue their own service requests to be picked up by the
    actual top-level catch-all cosim service generator."""

    clk = Clock()
    rst = Input(Bits(1))

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)

      mmio_read = esi.FuncService.get_coerced(esi.AppID("__cosim_mmio_read"),
                                              esi.MMIO.read.type)
      ChannelMMIO(esi.MMIO,
                  appid=esi.AppID("__cosim_mmio"),
                  clk=ports.clk,
                  rst=ports.rst,
                  read=mmio_read)

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
