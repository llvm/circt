# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import (Clock, Input, Module, System, generator)
from pycde import esi
from pycde.common import AppID, Output
from pycde.constructs import Wire
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection)
from pycde.testing import unittestmodule
from pycde.signals import BitsSignal, ChannelSignal

from typing import Dict

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.FROM, Bits(16)),
    BundledChannel("req", ChannelDirection.TO, Bits(24))
])


@esi.ServiceDecl
class HostComms:
  req_resp = TestBundle


class LoopbackInOut(Module):

  @generator
  def construct(self):
    loopback = Wire(Channel(Bits(16)))
    call_bundle = HostComms.req_resp(AppID("loopback_inout"))
    froms = call_bundle.unpack(resp=loopback)
    from_host = froms['req']
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
    System.current().platform = "cosim"

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


# CHECK-LABEL: hw.module @MultiplexerTop(in %clk : i1, in %rst : i1, in %trunk_in : i256, in %trunk_in_valid : i1, in %trunk_out_ready : i1, out trunk_in_ready : i1, out trunk_out : i256, out trunk_out_valid : i1) attributes {output_file = #hw.output_file<"MultiplexerTop.sv", includeReplicatedOps>} {
# CHECK:         %c0_i240 = hw.constant 0 : i240
# CHECK:         [[R0:%.+]] = comb.extract %trunk_in from 0 {sv.namehint = "trunk_in_0upto24"} : (i256) -> i24
# CHECK:         [[R1:%.+]] = comb.concat %c0_i240, %LoopbackInOut.loopback_inout_resp : i240, i16
# CHECK:         %LoopbackInOut.loopback_inout_req_ready, %LoopbackInOut.loopback_inout_resp, %LoopbackInOut.loopback_inout_resp_valid = hw.instance "LoopbackInOut" sym @LoopbackInOut @LoopbackInOut(loopback_inout_req: [[R0]]: i24, loopback_inout_req_valid: %trunk_in_valid: i1, loopback_inout_resp_ready: %trunk_out_ready: i1) -> (loopback_inout_req_ready: i1, loopback_inout_resp: i16, loopback_inout_resp_valid: i1)
# CHECK:         hw.instance "__manifest" @__ESIManifest() -> ()
# CHECK:         hw.output %LoopbackInOut.loopback_inout_req_ready, [[R1]], %LoopbackInOut.loopback_inout_resp_valid : i1, i256, i1
# CHECK:       }
# CHECK-LABEL: hw.module @LoopbackInOut(in %loopback_inout_req : i24, in %loopback_inout_req_valid : i1, in %loopback_inout_resp_ready : i1, out loopback_inout_req_ready : i1, out loopback_inout_resp : i16, out loopback_inout_resp_valid : i1) attributes {output_file = #hw.output_file<"LoopbackInOut.sv", includeReplicatedOps>} {
# CHECK:         [[R0:%.+]] = comb.extract %loopback_inout_req from 0 : (i24) -> i16
# CHECK:         hw.output %loopback_inout_resp_ready, [[R0]], %loopback_inout_req_valid : i1, i16, i1
# CHECK:       }
