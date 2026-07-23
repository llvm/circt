#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Optional, Tuple, Type

from pycde.common import AppID, Clock, Input, Output
from pycde.module import Module, generator
from pycde.support import get_user_loc
from pycde.system import System
from pycde.signals import ChannelSignal
from pycde.types import (Bits, Channel, StructType, UInt)
from pycde.constructs import Wire
from pycde import esi

from .common import (ChannelEngineService, ChannelHostMem, ChannelMMIO,
                     DesignResetController, ResetCycles)
from .dma import OneItemBuffersFromHost, OneItemBuffersToHost

from pycde.circt import ir
from pycde.circt.dialects import esi as raw_esi

from pathlib import Path

__root_dir__ = Path(__file__).parent.parent


def CosimBSP(
    user_module: Type[Module],
    dma_engine_pair: Optional[Tuple[Callable, Callable]] = None,
) -> Module:
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

    # Asserted for one cycle when the design requests a reset (via an MMIO write
    # to the header). Consumed by the top-level module which performs the reset.
    # NOTE: must be declared after 'hostmem_read'/'hostmem_write' so it does not
    # perturb their output port indices (they alias the sub-module's shared
    # Output objects, whose '.idx' is reused for the instance output lookup).
    reset_request = Output(Bits(1))

    @generator
    def build(ports):
      user_module(clk=ports.clk, rst=ports.rst)
      esi.TelemetryMMIO(esi.Telemetry,
                        appid=esi.AppID("__telemetry"),
                        clk=ports.clk,
                        rst=ports.rst)

      if dma_engine_pair is not None:
        ChannelEngineService(dma_engine_pair[0], dma_engine_pair[1])(
            None,
            appid=esi.AppID("__channel_engines"),
            clk=ports.clk,
            rst=ports.rst)

      mmio = ChannelMMIO(esi.MMIO,
                         appid=esi.AppID("__cosim_mmio"),
                         clk=ports.clk,
                         rst=ports.rst,
                         cmd=ports.mmio)
      ports.reset_request = mmio.reset_request

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

      mmio_read_write = esi.FuncService.get(
          esi.AppID("__cosim_mmio_read_write"), esi.MMIO.read_write.type)

      # The design can request a reset via an MMIO write to the header. Once
      # requested, reset the entire design (but not the cosim link) after a
      # fixed number of cycles. The reset controller is driven by the external
      # reset only so its countdown is not disturbed by the reset it generates.
      design_reset = Wire(Bits(1))
      reset_pending = Wire(Bits(1))
      combined_rst = ports.rst | design_reset

      # Once a reset has been requested, stop accepting *new* MMIO transactions
      # so that none is in flight when the design (including the MMIO plane) is
      # actually reset 'ResetCycles' cycles later. Any already-accepted
      # transaction has that window to drain; a command held here is released
      # once the reset completes and the MMIO plane is back up. The gate is
      # purely combinational off the (registered) 'reset_pending', so it blocks
      # at a clean cycle boundary: when pending is high the command is not
      # presented downstream and the host sees back-pressure (so it retains the
      # command), and neither side completes a handshake. This logic lives in
      # the external-reset domain so it is not cleared by the reset it guards.
      mmio_resp = Wire(Channel(esi.MMIODataType))
      host_cmd = mmio_read_write.unpack(data=mmio_resp)['cmd']
      gate_ready = Wire(Bits(1))
      cmd_payload, cmd_valid = host_cmd.unwrap(gate_ready)
      fwd_valid = (cmd_valid & ~reset_pending).as_bits()
      gated_cmd, fwd_ready = host_cmd.type.wrap(cmd_payload, fwd_valid)
      gate_ready.assign((fwd_ready & ~reset_pending).as_bits())
      gated_mmio, gated_froms = esi.MMIO.read_write.type.pack(cmd=gated_cmd)
      mmio_resp.assign(gated_froms['data'])

      wrapper = ESI_Cosim_UserTopWrapper(clk=ports.clk,
                                         rst=combined_rst,
                                         mmio=gated_mmio)
      reset_controller = DesignResetController(ResetCycles)(
          clk=ports.clk, rst=ports.rst, reset_request=wrapper.reset_request)
      design_reset.assign(reset_controller.design_reset)
      reset_pending.assign(reset_controller.reset_pending)

      # While a reset is pending, drop host-driven responses to the design's
      # hostmem service. In-flight requests issued before the reset will have
      # their responses arrive after the design (and its hostmem tag/demux
      # state) has been reset; delivering them would corrupt the fresh state or
      # hang on a response the reset design never consumes. Draining them here
      # (assert ready toward the host, never present a valid downstream) keeps
      # the host response channels flowing. This lives in the external-reset
      # domain so it is not cleared by the reset it guards.
      def drop_while_resetting(chan: ChannelSignal) -> ChannelSignal:
        src_ready = Wire(Bits(1))
        payload, valid = chan.unwrap(src_ready)
        fwd_valid = (valid & ~reset_pending).as_bits()
        gated, dn_ready = chan.type.wrap(payload, fwd_valid)
        src_ready.assign((dn_ready | reset_pending).as_bits())
        return gated

      resp_channel = esi.ChannelService.from_host(
          esi.AppID("__cosim_hostmem_read_resp"),
          StructType([
              ("tag", UInt(8)),
              ("data", Bits(ESI_Cosim_UserTopWrapper.HostMemWidth)),
              ("last", Bits(1)),
          ]))
      req = wrapper.hostmem_read.unpack(
          resp=drop_while_resetting(resp_channel))['req']
      esi.ChannelService.to_host(esi.AppID("__cosim_hostmem_read_req"), req)

      ack_wire = Wire(Channel(UInt(8)))
      write_req = wrapper.hostmem_write.unpack(ackTag=ack_wire)['req']
      ack_tag = esi.CallService.call(esi.AppID("__cosim_hostmem_write"),
                                     write_req, UInt(8))
      ack_wire.assign(drop_while_resetting(ack_tag))

      cosim_svc = raw_esi.ServiceInstanceOp(
          result=[],
          appID=AppID("cosim")._appid,
          service_symbol=None,
          impl_type=ir.StringAttr.get("cosim"),
          inputs=[ports.clk.value, ports.rst.value],
          loc=get_user_loc())
      core_freq = System.current().core_freq
      if core_freq is not None:
        cosim_svc.operation.attributes[
            "esi.core_clock_frequency_hz"] = ir.IntegerAttr.get(
                ir.IntegerType.get_unsigned(64), core_freq)

  return ESI_Cosim_Top


def CosimBSP_DMA(user_module: Type[Module]) -> Module:
  return CosimBSP(user_module,
                  dma_engine_pair=(OneItemBuffersToHost,
                                   OneItemBuffersFromHost))
