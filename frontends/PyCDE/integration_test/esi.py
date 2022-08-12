# REQUIRES: esi-cosim
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --schema %t/schema.capnp %s %t/*.sv
# PY: import support.loopback as test
# PY: rpc = test.LoopbackTester(rpcschemapath, simhostport)
# PY: rpc.test_two_chan_loopback(25)
# PY: rpc.test_one_chan_loopback(25)

import pycde
from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi
from pycde.constructs import Wire

import sys


@esi.ServiceDecl
class HostComms:
  to_host = esi.ToServer(types.any)
  from_host = esi.FromServer(types.any)
  req_resp = esi.ToFromServer(to_server_type=types.i16,
                              to_client_type=types.i32)


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


@module
class LoopbackInOut:

  @generator
  def construct(ports):
    loopback = Wire(types.channel(types.i16))
    from_host = HostComms.req_resp("loopback_inout", loopback)
    ready = Wire(types.i1)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


@module
class top:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)

    LoopbackInOut()

    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)


s = pycde.System([top], name="ESILoopback", output_directory=sys.argv[1])
s.print()
s.generate()
s.emit_outputs()
