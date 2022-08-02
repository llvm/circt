# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --schema %t/schema.capnp %s %t/*.sv
# PY: import support.loopback as test
# PY: rpc = test.LoopbackTester(rpcschemapath, simhostport)
# PY: rpc.test_two_chan_loopback(25)

import pycde
from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi

import sys


@module
class Producer:
  clk = Input(types.i1)
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = esi.HostComms.from_host(types.i32, "loopback_in")
    ports.int_out = chan


@module
class Consumer:
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    esi.HostComms.to_host(ports.int_in, "loopback_out")


@module
class top:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(esi.HostComms, ports.clk, ports.rst)


s = pycde.System([top], name="ESILoopback", output_directory=sys.argv[1])
s.print()
s.generate()
s.emit_outputs()
