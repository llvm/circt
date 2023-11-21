# XFAIL: *
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
from pycde.signals import BitVectorSignal, ChannelSignal

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
    loopback = Wire(types.channel(types.i16))
    call_bundle, froms = TestBundle.pack(resp=loopback)
    from_host = froms['req']
    HostComms.req_resp(call_bundle, AppID("loopback_inout", 0))
    ready = Wire(types.i1)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class MultiplexerService(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(types.i1)

  # Underlying channel is an untyped, 256-bit LI channel.
  trunk_in = Input(types.i256)
  trunk_in_valid = Input(types.i1)
  trunk_in_ready = Output(types.i1)
  trunk_out = Output(types.i256)
  trunk_out_valid = Output(types.i1)
  trunk_out_ready = Input(types.i1)

  @generator
  def generate(self, bundles):

    input_reqs = channels.to_server_reqs
    if len(input_reqs) > 1:
      raise Exception("Multiple to_server requests not supported")
    MultiplexerService.unwrap_and_pad(self, input_reqs[0])

    output_reqs = channels.to_client_reqs
    if len(output_reqs) > 1:
      raise Exception("Multiple to_client requests not supported")
    output_req = output_reqs[0]
    output_chan, ready = MultiplexerService.slice_and_wrap(
        self, output_req.type)
    output_req.assign(output_chan)
    self.trunk_in_ready = ready

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
    assert isinstance(data, BitVectorSignal)
    assert len(data) <= 256
    ports.trunk_out = data.pad_or_truncate(256)
    ports.trunk_out_valid = valid


@unittestmodule(run_passes=True, print_after_passes=True, emit_outputs=True)
class MultiplexerTop(Module):
  clk = Clock()
  rst = Input(types.i1)

  trunk_in = Input(types.i256)
  trunk_in_valid = Input(types.i1)
  trunk_in_ready = Output(types.i1)
  trunk_out = Output(types.i256)
  trunk_out_valid = Output(types.i1)
  trunk_out_ready = Input(types.i1)

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


class PassUpService(esi.ServiceImplementation):

  @generator
  def generate(self, channels):
    for req in channels.to_server_reqs:
      name = "out_" + "_".join(req.client_name)
      esi.PureModule.output_port(name, req)
    for req in channels.to_client_reqs:
      name = "in_" + "_".join(req.client_name)
      req.assign(esi.PureModule.input_port(name, req.type))


# CHECK-LABEL:  hw.module @PureTest<FOO: i5, STR: none>(in %in_Producer_loopback_in : i32, in %in_Producer_loopback_in_valid : i1, in %in_prod2_loopback_in : i32, in %in_prod2_loopback_in_valid : i1, in %clk : i1, in %out_Consumer_loopback_out_ready : i1, in %p2_int_ready : i1, out in_Producer_loopback_in_ready : i1, out in_prod2_loopback_in_ready : i1, out out_Consumer_loopback_out : i32, out out_Consumer_loopback_out_valid : i1, out p2_int : i32, out p2_int_valid : i1)
# CHECK-NEXT:     %Producer.loopback_in_ready, %Producer.int_out, %Producer.int_out_valid = hw.instance "Producer" sym @Producer @Producer{{.*}}(clk: %clk: i1, loopback_in: %in_Producer_loopback_in: i32, loopback_in_valid: %in_Producer_loopback_in_valid: i1, int_out_ready: %Consumer.int_in_ready: i1) -> (loopback_in_ready: i1, int_out: i32, int_out_valid: i1)
# CHECK-NEXT:     %Consumer.int_in_ready, %Consumer.loopback_out, %Consumer.loopback_out_valid = hw.instance "Consumer" sym @Consumer @Consumer{{.*}}(clk: %clk: i1, int_in: %Producer.int_out: i32, int_in_valid: %Producer.int_out_valid: i1, loopback_out_ready: %out_Consumer_loopback_out_ready: i1) -> (int_in_ready: i1, loopback_out: i32, loopback_out_valid: i1)
# CHECK-NEXT:     %prod2.loopback_in_ready, %prod2.int_out, %prod2.int_out_valid = hw.instance "prod2" sym @prod2 @Producer{{.*}}(clk: %clk: i1, loopback_in: %in_prod2_loopback_in: i32, loopback_in_valid: %in_prod2_loopback_in_valid: i1, int_out_ready: %p2_int_ready: i1) -> (loopback_in_ready: i1, int_out: i32, int_out_valid: i1)
# CHECK-NEXT:     hw.output %Producer.loopback_in_ready, %prod2.loopback_in_ready, %Consumer.loopback_out, %Consumer.loopback_out_valid, %prod2.int_out, %prod2.int_out_valid : i1, i1, i32, i1, i32, i1
@unittestmodule(run_passes=True, print_after_passes=True, emit_outputs=True)
class PureTest(esi.PureModule):

  @generator
  def construct(ports):
    PassUpService(None)

    clk = esi.PureModule.input_port("clk", ClockType())
    p = Producer(clk=clk)
    Consumer(clk=clk, int_in=p.int_out)
    p2 = Producer(clk=clk, instance_name="prod2")
    esi.PureModule.output_port("p2_int", p2.int_out)
    esi.PureModule.param("FOO", Bits(5))
    esi.PureModule.param("STR")


ExStruct = types.struct({
    'a': Bits(4),
    'b': UInt(32),
})
