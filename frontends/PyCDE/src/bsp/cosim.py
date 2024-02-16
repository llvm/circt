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

from .common import AxiMMIO

from ..circt import ir
from ..circt.dialects import esi as raw_esi

from pathlib import Path

__root_dir__ = Path(__file__).parent.parent


class Cosim_MMIO(Module):
  """External module backed by DPI calls into the coimulation driver. Provides
  an AXI-lite interface. An emulator."""

  clk = Clock()
  rst = Input(Bits(1))

  # MMIO read: address channel.
  arvalid = Output(Bits(1))
  arready = Input(Bits(1))
  araddr = Output(Bits(32))

  # MMIO read: data response channel.
  rvalid = Input(Bits(1))
  rready = Output(Bits(1))
  rdata = Input(Bits(32))
  rresp = Input(Bits(2))

  # MMIO write: address channel.
  awvalid = Output(Bits(1))
  awready = Input(Bits(1))
  awaddr = Output(Bits(32))

  # MMIO write: data channel.
  wvalid = Output(Bits(1))
  wready = Input(Bits(1))
  wdata = Output(Bits(32))

  # MMIO write: write response channel.
  bvalid = Input(Bits(1))
  bready = Output(Bits(1))
  bresp = Input(Bits(2))


def CosimBSP(user_module):
  """Wrap and return a cosimulation 'board support package' containing
  'user_module'"""

  class ESI_Cosim_Top(Module):
    clk = Clock()
    rst = Input(Bits(1))

    @generator
    def build(ports):
      System.current().platform = "cosim"

      user_module(clk=ports.clk, rst=ports.rst)
      raw_esi.ServiceInstanceOp(result=[],
                                appID=AppID("cosim")._appid,
                                service_symbol=None,
                                impl_type=ir.StringAttr.get("cosim"),
                                inputs=[ports.clk.value, ports.rst.value])

      # Instantiate both the Cosim MMIO emulator and the AXI-lite MMIO service
      # implementation and wire them together. The CosimMMIO emulator has a
      # 32-bit address whereas the AXI-lite MMIO service implementation has a
      # 20-bit address. Other than that, the ports are the same so use some
      # PyCDE magic to wire them together.
      cosim_mmio_wire_inputs = {
          port.name: Wire(port.type)
          for port in Cosim_MMIO.inputs()
          if port.name != "clk" and port.name != "rst"
      }
      cosim_mmio = Cosim_MMIO(clk=ports.clk,
                              rst=ports.rst,
                              **cosim_mmio_wire_inputs)

      axi_mmio_inputs = cosim_mmio.outputs()
      axi_mmio_inputs["araddr"] = axi_mmio_inputs["araddr"][0:20]
      axi_mmio = AxiMMIO(esi.MMIO,
                         appid=AppID("mmio"),
                         clk=ports.clk,
                         rst=ports.rst,
                         **axi_mmio_inputs)
      for pn, s in axi_mmio.outputs().items():
        if pn == "awaddr":
          s = s.pad_or_truncate(32)
        cosim_mmio_wire_inputs[pn].assign(s)

  return ESI_Cosim_Top
