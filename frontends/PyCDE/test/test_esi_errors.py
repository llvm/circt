# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% py-split-input-file.py %s 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, Output, OutputChannel, Module,
                   generator)
from pycde.common import SendBundle, RecvBundle
from pycde.types import Bit, Bits, Bundle, BundledChannel, Channel, ChannelDirection
from pycde import esi
from pycde.testing import unittestmodule

BIT = Bit
I1 = Bit
I2 = Bits(2)
I4 = Bits(4)
I32 = Bits(32)

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
  int_out = OutputChannel(I32)

  @generator
  def construct(ports):
    chan = HostComms.from_host("loopback_in", I32)
    ports.int_out = chan


class Consumer(Module):
  clk = Clock()
  int_in = InputChannel(I32)

  @generator
  def construct(ports):
    HostComms.to_host(ports.int_in, "loopback_out")


@unittestmodule(print=True)
class LoopbackTop(Module):
  clk = Clock()
  rst = Input(BIT)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)


# -----

Bundle1 = Bundle([
    BundledChannel("req", ChannelDirection.TO, Channel(I32)),
    BundledChannel("resp", ChannelDirection.FROM, Channel(BIT)),
])


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class SendBundleTest(Module):
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(I32)
  i1_out = OutputChannel(BIT)

  @generator
  def build(self):
    (self.b_send, from_chans) = Bundle1.pack()
    # CHECK: ValueError: Missing channels: req


# -----


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class SendBundleTest(Module):
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(I32)
  i1_out = OutputChannel(BIT)

  @generator
  def build(self):
    (self.b_send, from_chans) = Bundle1.pack(asdf=self.s1_in)
    # CHECK: ValueError: Unknown channel name 'asdf'


# -----


@unittestmodule()
class SendBundleTest(Module):
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(I2)

  @generator
  def build(self):
    (self.b_send, from_chans) = Bundle1.pack(req=self.s1_in)
    # CHECK: TypeError: Expected channel type Channel<Bits<32>, ValidReady>, got Channel<Bits<2>, ValidReady> on channel 'req'


# -----


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class RecvBundleTest(Module):
  b_recv = RecvBundle(Bundle1)
  s1_out = OutputChannel(I32)
  i1_in = InputChannel(BIT)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack()
    # CHECK: ValueError: Missing channel values for resp


# -----


@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class RecvBundleTest(Module):
  b_recv = RecvBundle(Bundle1)
  s1_out = OutputChannel(I32)
  i1_in = InputChannel(BIT)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack(asdf=self.i1_in)
    # CHECK: ValueError: Unknown channel name 'asdf'


# -----


@unittestmodule()
class RecvBundleTest(Module):
  b_recv = RecvBundle(Bundle1)
  i1_in = InputChannel(I4)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack(resp=self.i1_in)
    # CHECK: TypeError: Expected channel type Channel<Bits<1>, ValidReady>, got Channel<Bits<4>, ValidReady> on channel 'resp'


# -----

try:
  Bundle([
      BundledChannel("req", ChannelDirection.TO, Channel(Bits(32))),
      BundledChannel("resp", ChannelDirection.TO, Channel(Bits(8))),
  ]).get_to_from()
  # CHECK: Bundle must have one channel in each direction.
except ValueError as e:
  print(e)

# -----

try:
  Bundle([
      BundledChannel("req", ChannelDirection.FROM, Channel(Bits(32))),
      BundledChannel("resp", ChannelDirection.FROM, Channel(Bits(8))),
  ]).get_to_from()
  # CHECK: Bundle must have one channel in each direction.
except ValueError as e:
  print(e)

# -----


@unittestmodule()
class CoerceBundleTransformWrongToWidth(Module):
  b_in = Input(
      Bundle([
          BundledChannel("req", ChannelDirection.TO, Channel(Bits(32))),
          BundledChannel("resp", ChannelDirection.FROM, Channel(Bits(8))),
      ]))
  b_out = Output(
      Bundle([
          BundledChannel("arg", ChannelDirection.TO, Channel(Bits(24))),
          BundledChannel("result", ChannelDirection.FROM, Channel(Bits(16))),
      ]))

  @generator
  def build(ports):
    ports.b_out = ports.b_in.coerce(
        CoerceBundleTransformWrongToWidth.b_out.type, lambda x: x[0:30],
        lambda x: x[0:8])
    # CHECK: TypeError: Expected channel type Channel<Bits<24>, ValidReady>, got Channel<Bits<30>, ValidReady> on TO channel


# -----


@unittestmodule()
class CoerceBundleTransformWrongFromWidth(Module):
  b_in = Input(
      Bundle([
          BundledChannel("req", ChannelDirection.TO, Channel(Bits(32))),
          BundledChannel("resp", ChannelDirection.FROM, Channel(Bits(8))),
      ]))
  b_out = Output(
      Bundle([
          BundledChannel("arg", ChannelDirection.TO, Channel(Bits(24))),
          BundledChannel("result", ChannelDirection.FROM, Channel(Bits(16))),
      ]))

  @generator
  def build(ports):
    ports.b_out = ports.b_in.coerce(
        CoerceBundleTransformWrongFromWidth.b_out.type, lambda x: x[0:24],
        lambda x: x[0:10])
    # CHECK: TypeError: Expected channel type Channel<Bits<8>, ValidReady>, got Channel<Bits<10>, ValidReady> on FROM channel
