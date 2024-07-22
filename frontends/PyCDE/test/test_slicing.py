# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import System, Input, Output, generator, Module
from pycde.types import Bits, dim

# CHECK-LABEL: hw.module @MyMod(in %in_port : i8, out out0 : i5, out out1 : i5, out out2 : i0)
# CHECK:         %0 = comb.extract %in_port from 3 {sv.namehint = "in_port_3upto8"} : (i8) -> i5
# CHECK:         %1 = comb.extract %in_port from 0 {sv.namehint = "in_port_0upto5"} : (i8) -> i5
# CHECK:         %c0_i0 = hw.constant 0 : i0
# CHECK:         hw.output %0, %1, %c0_i0 : i5, i5, i0
# CHECK:       }


class MyMod(Module):
  in_port = Input(dim(8))
  out0 = Output(dim(5))
  out1 = Output(dim(5))
  out2 = Output(Bits(0))

  @generator
  def construct(mod):
    # Partial lower slice
    mod.out0 = mod.in_port[3:]
    # partial upper slice
    mod.out1 = mod.in_port[:5]
    # empty slice
    mod.out2 = mod.in_port[1:1]


top = System([MyMod])
top.generate()
top.print()
