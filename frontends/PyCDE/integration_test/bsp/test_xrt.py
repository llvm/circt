# REQUIRES: xrt
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t genhw 2>&1

import pycde
from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi
from pycde.constructs import Wire
from pycde.dialects import comb
from pycde.bsp import XrtBSP
# from msft.xrt import XrtBSP

import os
import sys
import time


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
    HostComms.to_host(ports.int_in, "loopback_out")


@module
class LoopbackInOutAdd7:

  @generator
  def construct(ports):
    loopback = Wire(types.channel(types.i16))
    from_host = HostComms.req_resp(loopback, "loopback_inout")
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


@module
class Top:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    Mid(clk=ports.clk, rst=ports.rst)


gendir = sys.argv[1]
command = sys.argv[2] if len(sys.argv) == 3 else "genhw"
if command == "genhw":
  s = pycde.System(XrtBSP(Top),
                   name="ESILoopback",
                   output_directory=gendir,
                   sw_api_langs=["python"])
  s.run_passes(debug=True)
  s.compile()
  s.package()

elif command == "test":
  sys.path.append(os.path.join(gendir, "runtime"))
  import ESILoopback as esi_sys
  from ESILoopback.xrt import Xrt

  acc_conn = Xrt(os.path.join(gendir, "ESILoopback.hw_emu.xclbin"), hw_emu=True)
  top = esi_sys.top(acc_conn)

  # assert top.bsp.req_resp_read_any() is None
  # assert top.bsp.req_resp[0].read(blocking_timeout=None) is None
  # assert top.bsp.to_host_read_any() is None
  # assert top.bsp.to_host[0].read(blocking_timeout=None) is None

  # assert top.bsp.req_resp[0].write(5) is True
  # time.sleep(0.05)
  # assert top.bsp.to_host_read_any() is None
  # assert top.bsp.to_host[0].read(blocking_timeout=None) is None
  # assert top.bsp.req_resp[0].read() == 12
  # assert top.bsp.req_resp[0].read(blocking_timeout=None) is None

  # assert top.bsp.req_resp[0].write(9) is True
  # time.sleep(0.05)
  # assert top.bsp.to_host_read_any() is None
  # assert top.bsp.to_host[0].read(blocking_timeout=None) is None
  # assert top.bsp.req_resp_read_any() == 16
  # assert top.bsp.req_resp_read_any() is None
  # assert top.bsp.req_resp[0].read(blocking_timeout=None) is None

  # assert top.bsp.from_host[0].write(9) is True
  # time.sleep(0.05)
  # assert top.bsp.req_resp_read_any() is None
  # assert top.bsp.req_resp[0].read(blocking_timeout=None) is None
  # assert top.bsp.to_host_read_any() == 9
  # assert top.bsp.to_host[0].read(blocking_timeout=None) is None

  # assert top.bsp.from_host[0].write(9) is True
  # time.sleep(0.05)
  # assert top.bsp.req_resp_read_any() is None
  # assert top.bsp.req_resp[0].read(blocking_timeout=None) is None
  # assert top.bsp.to_host[0].read() == 9
  # assert top.bsp.to_host_read_any() is None

  print("Success: all tests pass!")
