# REQUIRES: esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --tmpdir %t --schema %t/schema.capnp %s %t/*.sv
# PY: from esi_test import run_cosim
# PY: run_cosim(tmpdir, rpcschemapath, simhostport)

import pycde
from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi
from pycde.constructs import Wire
from pycde.dialects import comb

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
class LoopbackInOutAdd7:

  @generator
  def construct(ports):
    loopback = Wire(types.channel(types.i16))
    from_host = HostComms.req_resp("loopback_inout", loopback)
    ready = Wire(types.i1)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    # TODO: clean this up with PyCDE overloads (they're currently a little bit
    # broken for this use-case).
    data = comb.AddOp(data, types.i16(7))
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


@module
class Mid:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)

    LoopbackInOutAdd7()

    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)


@module
class top:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    Mid(clk=ports.clk, rst=ports.rst)


if __name__ == "__main__":
  s = pycde.System([top], name="ESILoopback", output_directory=sys.argv[1])
  s.generate()
  s.emit_outputs()
  s.build_api("python")


def run_cosim(tmpdir, schema_path, rpchostport):
  sys.path.append(tmpdir)
  import esi_rt.ESILoopback as esi_sys
  from esi_rt.common import Cosim

  top = esi_sys.top(Cosim(schema_path, rpchostport))

  assert top.mid.host_comms.req_resp_read_any() is None
  assert top.mid.host_comms.req_resp[0].read(blocking_timeout=None) is None
  assert top.mid.host_comms.to_host_read_any() is None
  assert top.mid.host_comms.to_host[0].read(blocking_timeout=None) is None

  assert top.mid.host_comms.req_resp[0].write(5) is True
  assert top.mid.host_comms.to_host_read_any() is None
  assert top.mid.host_comms.to_host[0].read(blocking_timeout=None) is None
  assert top.mid.host_comms.req_resp[0].read() == 12
  assert top.mid.host_comms.req_resp[0].read(blocking_timeout=None) is None

  assert top.mid.host_comms.req_resp[0].write(9) is True
  assert top.mid.host_comms.to_host_read_any() is None
  assert top.mid.host_comms.to_host[0].read(blocking_timeout=None) is None
  assert top.mid.host_comms.req_resp_read_any() == 16
  assert top.mid.host_comms.req_resp_read_any() is None
  assert top.mid.host_comms.req_resp[0].read(blocking_timeout=None) is None

  assert top.mid.host_comms.from_host[0].write(9) is True
  assert top.mid.host_comms.req_resp_read_any() is None
  assert top.mid.host_comms.req_resp[0].read(blocking_timeout=None) is None
  assert top.mid.host_comms.to_host_read_any() == 9
  assert top.mid.host_comms.to_host[0].read(blocking_timeout=None) is None

  assert top.mid.host_comms.from_host[0].write(9) is True
  assert top.mid.host_comms.req_resp_read_any() is None
  assert top.mid.host_comms.req_resp[0].read(blocking_timeout=None) is None
  assert top.mid.host_comms.to_host[0].read() == 9
  assert top.mid.host_comms.to_host_read_any() is None

  print("Success: all tests pass!")
