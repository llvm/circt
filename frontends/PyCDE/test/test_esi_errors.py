# RUN: rm -rf %t
# RUN: %PYTHON% py-split-input-file.py %s 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, OutputChannel, Module, generator,
                   types)
from pycde.common import SendBundle, RecvBundle
from pycde.types import Bits, Bundle, BundledChannel, ChannelDirection
from pycde import esi
from pycde.testing import unittestmodule

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.TO, Bits(16)),
    BundledChannel("req", ChannelDirection.FROM, Bits(24))
])

TestFromBundle = Bundle(
    [BundledChannel("ch1", ChannelDirection.FROM, Bits(32))])

TestToBundle = Bundle([BundledChannel("ch1", ChannelDirection.TO, Bits(32))])


@esi.ServiceDecl
class HostComms:
  req_resp = TestBundle
  from_host = TestFromBundle


class Producer(Module):
  clk = Clock()
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = HostComms.from_host("loopback_in", types.i32)
    ports.int_out = chan


class Consumer(Module):
  clk = Clock()
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    HostComms.to_host(ports.int_in, "loopback_out")


@unittestmodule(print=True)
class LoopbackTop(Module):
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)


class MultiplexerService(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(types.i1)

  def __init__(self, **inputs):
    super().__init__(HostComms, **inputs)

  @generator
  def generate(ports, channels):

    c = types.i128(0)
    v = types.i1(0)
    chan, rdy = types.channel(types.i128).wrap(c, v)
    try:
      # CHECKx: Channel type mismatch. Expected Channel<Bits<32>, ValidReady>, got Channel<Bits<128>, ValidReady>.
      channels.to_client_reqs[0].assign(chan)
    except Exception as e:
      print(e)
    try:
      input_req = channels.to_server_reqs[0]
      channels.to_client_reqs[1].assign(input_req)
      # CHECKx: Producer_1.loopback_in has already been connected.
      channels.to_client_reqs[1].assign(input_req)
    except Exception as e:
      print(e)
    # CHECKx: ValueError: Producer.loopback_in has not been connected.


# TODO: Re-enable this test once service generation is fixed.
# @unittestmodule(run_passes=True, print_after_passes=True)
class MultiplexerTop(Module):
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def construct(ports):
    MultiplexerService(clk=ports.clk, rst=ports.rst)

    p = Producer(clk=ports.clk)
    p1 = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)


# -----


class BrokenService(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(types.i1)

  def __init__(self, **inputs):
    super().__init__(HostComms, **inputs)

  @generator
  def generate(ports, channels):
    return "asdf"
    # CHECKx: ValueError: Generators must a return a bool or None


# TODO: Re-enable this test once service generation is fixed.
# @unittestmodule(run_passes=True, print_after_passes=True)
class BrokenTop(Module):
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def construct(ports):
    BrokenService(clk=ports.clk, rst=ports.rst)


# -----

Bundle1 = Bundle([
    BundledChannel("req", ChannelDirection.TO, types.channel(types.i32)),
    BundledChannel("resp", ChannelDirection.FROM, types.channel(types.i1)),
])


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class SendBundleTest(Module):
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(types.i32)
  i1_out = OutputChannel(types.i1)

  @generator
  def build(self):
    (self.b_send, from_chans) = Bundle1.pack()
    # CHECK: ValueError: Missing channels: req


# -----


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class SendBundleTest(Module):
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(types.i32)
  i1_out = OutputChannel(types.i1)

  @generator
  def build(self):
    (self.b_send, from_chans) = Bundle1.pack(asdf=self.s1_in)
    # CHECK: ValueError: Unknown channel name 'asdf'


# -----


@unittestmodule()
class SendBundleTest(Module):
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(types.i2)

  @generator
  def build(self):
    (self.b_send, from_chans) = Bundle1.pack(req=self.s1_in)
    # CHECK: TypeError: Expected channel type Channel<Bits<32>, ValidReady>, got Channel<Bits<2>, ValidReady> on channel 'req'


# -----


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class RecvBundleTest(Module):
  b_recv = RecvBundle(Bundle1)
  s1_out = OutputChannel(types.i32)
  i1_in = InputChannel(types.i1)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack()
    # CHECK: ValueError: Missing channel values for resp


# -----


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class RecvBundleTest(Module):
  b_recv = RecvBundle(Bundle1)
  s1_out = OutputChannel(types.i32)
  i1_in = InputChannel(types.i1)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack(asdf=self.i1_in)
    # CHECK: ValueError: Unknown channel name 'asdf'


# -----


@unittestmodule()
class RecvBundleTest(Module):
  b_recv = RecvBundle(Bundle1)
  i1_in = InputChannel(types.i4)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack(resp=self.i1_in)
    # CHECK: TypeError: Expected channel type Channel<Bits<1>, ValidReady>, got Channel<Bits<4>, ValidReady> on channel 'resp'
