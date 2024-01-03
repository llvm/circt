# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, OutputChannel, Module, generator,
                   types)
from pycde import esi
from pycde.common import AppID, Output, RecvBundle, SendBundle
from pycde.constructs import Wire
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, ChannelSignaling, UInt, ClockType)
from pycde.testing import unittestmodule
from pycde.signals import BitsSignal, ChannelSignal

from typing import Dict, List

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.TO, Bits(16)),
    BundledChannel("req", ChannelDirection.FROM, Bits(24))
])


@esi.ServiceDecl
class HostComms:
  req_resp = TestBundle


class LoopbackInOut(Module):

  @generator
  def construct(self):
    loopback = Wire(Channel(Bits(16)))
    call_bundle, froms = TestBundle.pack(resp=loopback)
    from_host = froms['req']
    HostComms.req_resp(call_bundle, AppID("loopback_inout", 0))
    ready = Wire(Bits(1))
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class MultiplexerService(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(Bits(1))

  # Underlying channel is an untyped, 256-bit LI channel.
  trunk_in = Input(Bits(256))
  trunk_in_valid = Input(Bits(1))
  trunk_in_ready = Output(Bits(1))
  trunk_out = Output(Bits(256))
  trunk_out_valid = Output(Bits(1))
  trunk_out_ready = Input(Bits(1))

  @generator
  def generate(self, bundles: esi._ServiceGeneratorBundles):
    assert len(
        bundles.to_client_reqs) == 1, "Only one connection request supported"
    bundle = bundles.to_client_reqs[0]
    to_req_types = {}
    for bundled_chan in bundle.type.channels:
      if bundled_chan.direction == ChannelDirection.TO:
        to_req_types[bundled_chan.name] = bundled_chan.channel

    to_channels = MultiplexerService._generate_to(self, to_req_types)
    bundle_sig, from_channels = bundle.type.pack(**to_channels)
    bundle.assign(bundle_sig)
    MultiplexerService._generate_from(self, from_channels)

  def _generate_from(self, from_reqs):
    if len(from_reqs) > 1:
      raise Exception("Multiple to_server requests not supported")
    for _, chan in from_reqs:
      MultiplexerService.unwrap_and_pad(self, chan)

  def _generate_to(
      self, to_req_types: Dict[str, Channel]) -> Dict[str, ChannelSignal]:
    if len(to_req_types) > 1:
      raise Exception("Multiple TO channels not supported")
    chan_name = list(to_req_types.keys())[0]
    output_type = to_req_types[chan_name]
    output_chan, ready = MultiplexerService.slice_and_wrap(self, output_type)
    self.trunk_in_ready = ready
    return {chan_name: output_chan}

  @staticmethod
  def slice_and_wrap(ports, channel_type: Channel):
    assert (channel_type.inner_type.width <= 256)
    sliced = ports.trunk_in[:channel_type.inner_type.width]
    return channel_type.wrap(sliced, ports.trunk_in_valid)

  @staticmethod
  def unwrap_and_pad(ports, input_channel: ChannelSignal):
    """
    Unwrap the input channel and pad it to 256 bits.
    """
    (data, valid) = input_channel.unwrap(ports.trunk_out_ready)
    assert isinstance(data, BitsSignal)
    assert len(data) <= 256
    ports.trunk_out = data.pad_or_truncate(256)
    ports.trunk_out_valid = valid


@unittestmodule(run_passes=True, print_after_passes=True, emit_outputs=True)
class MultiplexerTop(Module):
  clk = Clock()
  rst = Input(Bits(1))

  trunk_in = Input(Bits(256))
  trunk_in_valid = Input(Bits(1))
  trunk_in_ready = Output(Bits(1))
  trunk_out = Output(Bits(256))
  trunk_out_valid = Output(Bits(1))
  trunk_out_ready = Input(Bits(1))

  @generator
  def construct(ports):
    m = MultiplexerService(HostComms,
                           appid=AppID("mux", 0),
                           clk=ports.clk,
                           rst=ports.rst,
                           trunk_in=ports.trunk_in,
                           trunk_in_valid=ports.trunk_in_valid,
                           trunk_out_ready=ports.trunk_out_ready)

    ports.trunk_in_ready = m.trunk_in_ready
    ports.trunk_out = m.trunk_out
    ports.trunk_out_valid = m.trunk_out_valid

    LoopbackInOut()
