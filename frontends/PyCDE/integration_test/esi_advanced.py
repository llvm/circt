# REQUIRES: esi-runtime, esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim.py -- %PYTHON% %S/test_software/esi_advanced.py cosim env

import sys

from pycde import generator, Clock, Module, Reset, System
from pycde.bsp import get_bsp
from pycde.common import InputChannel, OutputChannel, Output
from pycde.types import Bits, UInt
from pycde import esi


class Merge(Module):
  clk = Clock()
  rst = Reset()
  a = InputChannel(UInt(8))
  b = InputChannel(UInt(8))

  x = OutputChannel(UInt(8))

  @generator
  def build(ports):
    chan = ports.a.type.merge(ports.a, ports.b)
    ports.x = chan


class Join(Module):
  clk = Clock()
  rst = Reset()
  a = InputChannel(UInt(8))
  b = InputChannel(UInt(8))

  x = OutputChannel(UInt(9))

  @generator
  def build(ports):
    joined = ports.a.type.join(ports.a, ports.b)
    ports.x = joined.transform(lambda x: x.a + x.b)


class Fork(Module):
  clk = Clock()
  rst = Reset()
  a = InputChannel(UInt(8))

  x = OutputChannel(UInt(8))
  y = OutputChannel(UInt(8))

  @generator
  def build(ports):
    x, y = ports.a.fork(ports.clk, ports.rst)
    ports.x = x
    ports.y = y


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    clk = ports.clk
    rst = ports.rst
    merge_a = esi.ChannelService.from_host(esi.AppID("merge_a"),
                                           UInt(8)).buffer(clk, rst, 1)
    merge_b = esi.ChannelService.from_host(esi.AppID("merge_b"),
                                           UInt(8)).buffer(clk, rst, 1)
    merge = Merge("merge_i8",
                  clk=ports.clk,
                  rst=ports.rst,
                  a=merge_a,
                  b=merge_b)
    esi.ChannelService.to_host(esi.AppID("merge_x"),
                               merge.x.buffer(clk, rst, 1))

    join_a = esi.ChannelService.from_host(esi.AppID("join_a"),
                                          UInt(8)).buffer(clk, rst, 1)
    join_b = esi.ChannelService.from_host(esi.AppID("join_b"),
                                          UInt(8)).buffer(clk, rst, 1)
    join = Join("join_i8", clk=ports.clk, rst=ports.rst, a=join_a, b=join_b)
    esi.ChannelService.to_host(
        esi.AppID("join_x"),
        join.x.buffer(clk, rst, 1).transform(lambda x: x.as_uint(16)))

    fork_a = esi.ChannelService.from_host(esi.AppID("fork_a"),
                                          UInt(8)).buffer(clk, rst, 1)
    fork = Fork("fork_i8", clk=ports.clk, rst=ports.rst, a=fork_a)
    esi.ChannelService.to_host(esi.AppID("fork_x"), fork.x.buffer(clk, rst, 1))
    esi.ChannelService.to_host(esi.AppID("fork_y"), fork.y.buffer(clk, rst, 1))


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = System(bsp(Top), name="ESIAdvanced", output_directory=sys.argv[1])
  s.generate()
  s.run_passes()
  s.compile()
  s.package()
