# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import module, generator, types, Input, Output
from pycde.constructs import If, Else, Then
import pycde

import sys


@module
class IfDemo:
  cond = Input(types.i1)
  out = Output(types.i8)

  @generator
  def build(ports):
    with If(ports.cond):
      with Then:
        v = types.i8(1)
      with Else:
        v = types.i8(0)
    ports.out = v


# CHECK-LABEL: msft.module @IfDemo {} (%cond: i1) -> (out: i8)
# CHECK:         %c1_i8 = hw.constant 1 : i8
# CHECK:         %c0_i8 = hw.constant 0 : i8
# CHECK:         [[r0:%.+]] = comb.mux %cond, %c0_i8, %c1_i8 : i8
# CHECK:         msft.output [[r0]] : i8

t = pycde.System([IfDemo], name="BehavioralTest", output_directory=sys.argv[1])
t.generate()
t.print()
t.emit_outputs()
