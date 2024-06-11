# REQUIRES: esi-runtime, esi-cosim, rtl-sim, esitester
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim.py -- esitester cosim env | FileCheck %s

import pycde
from pycde import AppID, Clock, Module, Reset, generator
from pycde.bsp import cosim
from pycde.constructs import Wire
from pycde.esi import CallService
from pycde.types import Bits, Channel, UInt

import sys


class PrintfExample(Module):
  """Call a printf function on the host once at startup."""

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    # CHECK: PrintfExample: 7
    arg_data = UInt(32)(7)

    sent_signal = Wire(Bits(1), "sent_signal")
    sent = Bits(1)(1).reg(ports.clk,
                          ports.rst,
                          ce=sent_signal,
                          rst_value=Bits(1)(0))
    arg_valid = ~sent & ~ports.rst
    arg_chan, arg_ready = Channel(UInt(32)).wrap(arg_data, arg_valid)
    sent_signal.assign(arg_ready & arg_valid)
    CallService.call(AppID("PrintfExample"), arg_chan, Bits(0))


class EsiTesterTop(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    PrintfExample(clk=ports.clk, rst=ports.rst)


if __name__ == "__main__":
  s = pycde.System(cosim.CosimBSP(EsiTesterTop),
                   name="EsiTester",
                   output_directory=sys.argv[1])
  s.compile()
  s.package()
