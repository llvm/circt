# RUN: %PYTHON% %s | FileCheck %s

from pycde import (Output, Input, module, generator, obj_to_value, types, dim, System, no_connect)
from pycde.module import externmodule


@module
class Taps:
  taps = Output(dim(8, 3))

  @generator
  def build(mod):
    return {"taps": [203, 100, 23]}


@externmodule
class StupidLegacy:
  ignore = Input(dim(1, 4))


class Top(System):
  BarType = types.struct({"foo": types.i12}, "bar")
  inputs = []
  outputs = []

  def build(self, top):
    obj_to_value({"foo": 7}, types.struct({"foo": types.i12}))
    obj_to_value([42, 45], dim(types.i8, 2))
    obj_to_value(5, types.i8)

    Top.BarType.create({"foo": 7})

    Taps()
    StupidLegacy(ignore=no_connect)


top = Top()
top.generate(["build"])
top.print()
# CHECK-LABEL: hw.module @top()
# CHECK:  %c7_i12 = hw.constant 7 : i12
# CHECK:  %0 = hw.struct_create (%c7_i12) : !hw.struct<foo: i12>
# CHECK:  %c42_i8 = hw.constant 42 : i8
# CHECK:  %c45_i8 = hw.constant 45 : i8
# CHECK:  %1 = hw.array_create %c45_i8, %c42_i8 : i8
# CHECK:  %c5_i8 = hw.constant 5 : i8
# CHECK:  %c7_i12_0 = hw.constant 7 : i12
# CHECK:  %2 = hw.struct_create (%c7_i12_0) : !hw.struct<foo: i12>
# CHECK:  %3 = hw.bitcast %2 : (!hw.struct<foo: i12>) -> !hw.typealias<@pycde::@bar, !hw.struct<foo: i12>>

# CHECK:  hw.module @pycde.Taps() -> (%taps: !hw.array<3xi8>)
# CHECK:    %c-53_i8 = hw.constant -53 : i8
# CHECK:    %c100_i8 = hw.constant 100 : i8
# CHECK:    %c23_i8 = hw.constant 23 : i8
# CHECK:    [[REG0:%.+]] = hw.array_create %c23_i8, %c100_i8, %c-53_i8 : i8
# CHECK:    hw.output [[REG0]] : !hw.array<3xi8>
