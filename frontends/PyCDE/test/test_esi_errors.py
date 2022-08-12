# RUN: rm -rf %t
# RUN: %PYTHON% py-split-input-file.py %s 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi
from pycde.testing import unittestmodule


@esi.ServiceDecl
class HostComms:
  to_host = esi.ToServer(types.any)
  from_host = esi.FromServer(types.any)


@module
class Producer:
  clk = Input(types.i1)
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = HostComms.from_host("loopback_in", types.i32)
    ports.int_out = chan


@module
class Consumer:
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    HostComms.to_host("loopback_out", ports.int_in)


@unittestmodule(print=True)
class LoopbackTop:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)


@esi.ServiceImplementation(HostComms)
class MultiplexerService:
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports, channels):

    c = types.i128(0)
    v = types.i1(0)
    chan, rdy = types.channel(types.i128).wrap(c, v)
    try:
      # CHECK: ChannelType mismatch. Expected channel<i32>, got channel<i128>.
      channels.to_client_reqs[0].assign(chan)
    except Exception as e:
      print(e)
    try:
      input_req = channels.to_server_reqs[0]
      channels.to_client_reqs[1].assign(input_req)
      # CHECK: Producer_1.loopback_in has already been connected.
      channels.to_client_reqs[1].assign(input_req)
    except Exception as e:
      print(e)
    # CHECK: ValueError: Producer.loopback_in has not been connected.


@unittestmodule(run_passes=True, print_after_passes=True)
class MultiplexerTop:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    MultiplexerService(clk=ports.clk, rst=ports.rst)

    p = Producer(clk=ports.clk)
    p1 = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)


# -----


@esi.ServiceImplementation(HostComms)
class BrokenService:
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports, channels):
    return "asdf"
    # CHECK: ValueError: Generators must a return a bool or None


@unittestmodule(run_passes=True, print_after_passes=True)
class BrokenTop:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    BrokenService(clk=ports.clk, rst=ports.rst)
