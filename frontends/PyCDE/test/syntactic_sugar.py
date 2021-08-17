# RUN: %PYTHON% %s | FileCheck %s

from pycde import (Output, Input, module, generator, obj_to_value, types, dim,
                   System, no_connect)
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


BarType = types.struct({"foo": types.i12}, "bar")


@module
class Top:

  @generator
  def build(_):
    obj_to_value({"foo": 7}, types.struct({"foo": types.i12}))
    obj_to_value([42, 45], dim(types.i8, 2))
    obj_to_value(5, types.i8)

    BarType.create({"foo": 7})

    Taps()
    StupidLegacy(ignore=no_connect)


@module
class ComplexPorts:
  clk = Input(types.i1)
  sel = Input(types.i2)
  data_in = Input(dim(32, 3))
  struct_data_in = Input(types.struct({"foo": types.i32}))

  a = Output(types.i32)
  b = Output(types.i32)
  c = Output(types.i32)

  @generator
  def build(mod):
    assert len(mod.data_in) == 3
    return {
        'a': mod.data_in[0].reg(mod.clk),
        'b': mod.data_in[mod.sel],
        'c': mod.struct_data_in.foo
    }


top = System([Top])
top.generate()
top.generate()
top.print()
# CHECK-LABEL: hw.module @pycde.Top()
# CHECK:  %c7_i12 = hw.constant 7 : i12
# CHECK:  %0 = hw.struct_create (%c7_i12) : !hw.struct<foo: i12>
# CHECK:  %c42_i8 = hw.constant 42 : i8
# CHECK:  %c45_i8 = hw.constant 45 : i8
# CHECK:  %1 = hw.array_create %c45_i8, %c42_i8 : i8
# CHECK:  %c5_i8 = hw.constant 5 : i8
# CHECK:  %c7_i12_0 = hw.constant 7 : i12
# CHECK:  %2 = hw.struct_create (%c7_i12_0) : !hw.typealias<@pycde::@bar, !hw.struct<foo: i12>>

# CHECK:  hw.module @pycde.Taps() -> (%taps: !hw.array<3xi8>)
# CHECK:    %c-53_i8 = hw.constant -53 : i8
# CHECK:    %c100_i8 = hw.constant 100 : i8
# CHECK:    %c23_i8 = hw.constant 23 : i8
# CHECK:    [[REG0:%.+]] = hw.array_create %c23_i8, %c100_i8, %c-53_i8 : i8
# CHECK:    hw.output [[REG0]] : !hw.array<3xi8>

sys = System([ComplexPorts])
sys.generate()
sys.print()
# CHECK:  hw.module @pycde.Comple_Ports(%clk: i1, %data_in: !hw.array<3xi32>, %sel: i2, %struct_data_in: !hw.struct<foo: i32>) -> (%a: i32, %b: i32, %c: i32) {
# CHECK:    %c0_i2 = hw.constant 0 : i2
# CHECK:    [[REG0:%.+]] = hw.array_get %data_in[%c0_i2] : !hw.array<3xi32>
# CHECK:    [[REGR:%.+]] = seq.compreg [[REG0]], %clk : i32
# CHECK:    [[REG1:%.+]] = hw.array_get %data_in[%sel] : !hw.array<3xi32>
# CHECK:    [[REG2:%.+]] = hw.struct_extract %struct_data_in["foo"] : !hw.struct<foo: i32>
# CHECK:    hw.output [[REGR]], [[REG1]], [[REG2]] : i32, i32, i32
