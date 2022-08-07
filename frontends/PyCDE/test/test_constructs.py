# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, types
from pycde.common import Clock, Input, Output
from pycde.constructs import Reg, Wire
from pycde.testing import unittestmodule


@unittestmodule()
class ComplexMux:
  clk = Clock()
  In = Input(types.i8)
  Out = Output(types.i8)
  OutReg = Output(types.i8)

  @generator
  def create(ports):
    w1 = Wire(types.i8)
    ports.Out = w1
    w1.assign(ports.In)

    r1 = Reg(types.i8)
    ports.Out2 = r1
    r1.assign(ports.In)
