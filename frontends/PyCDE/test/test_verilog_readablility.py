# RUN: %PYTHON% %s %t | FileCheck %s

from pycde import (Clock, Output, Input, Module, generator, dim, System)
from pycde.types import Bits

import sys


class WireNames(Module):
  clk = Clock()
  data_in = Input(dim(32, 3))
  sel = Input(Bits(2))

  a = Output(Bits(32))
  b = Output(Bits(32))

  @generator
  def build(self):
    foo = self.data_in[0]
    foo.name = "foo"
    arr_data = dim(32, 4)([1, 2, 3, 4], "arr_data")
    self.a = foo.reg(self.clk).reg(self.clk)
    self.b = arr_data[self.sel]


sys = System([WireNames], output_directory=sys.argv[1])
sys.generate()
sys.run_passes()
sys.print()
# CHECK-LABEL:  hw.module @WireNames(in %clk : i1, in %data_in : !hw.array<3xi32>, in %sel : i2, out a : i32, out b : i32)
# CHECK:    %foo__reg1 = sv.reg sym @foo__reg1 : !hw.inout<i32>
# CHECK:    %foo__reg2 = sv.reg sym @foo__reg2 : !hw.inout<i32>
# CHECK:    %{{.+}} = sv.read_inout %foo__reg2 : !hw.inout<i32>
# CHECK:    sv.alwaysff(posedge %clk)  {
# CHECK:      %c0_i2 = hw.constant 0 : i2
# CHECK:      %{{.+}} = hw.array_get %data_in[%c0_i2] {sv.namehint = "foo"} : !hw.array<3xi32>
# CHECK:      sv.passign %foo__reg1, %3 : i32
# CHECK:      %{{.+}} = sv.read_inout %foo__reg1 : !hw.inout<i32>
# CHECK:      sv.passign %foo__reg2, %4 : i32
# CHECK:    }
# CHECK:    %{{.+}} = hw.array_get %0[%sel] : !hw.array<4xi32>
# CHECK:    hw.output %1, %2 : i32, i32
# CHECK:  }
