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
  sel = Output(Bits(1))

  @generator
  def build(ports):
    chan, sel = ports.a.type.merge(ports.a, ports.b)
    ports.x = chan
    ports.sel = sel


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


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = System(bsp(Top), name="ESIAdvanced", output_directory=sys.argv[1])
  s.generate()
  s.run_passes()
  s.compile()
  s.package()
