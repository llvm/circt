# RUN: %PYTHON% %s | FileCheck %s

from pycde import (Output, Input, module, generator, types, dim, System,
                   no_connect)
from pycde.module import externmodule


@module
class Taps:
  taps = Output(dim(8, 3))

  @generator
  def build(ports):
    ports.taps = [203, 100, 23]


@externmodule
class StupidLegacy:
  ignore = Input(dim(1, 4))


BarType = types.struct({"foo": types.i12}, "bar")


@module
class Top:

  @generator
  def build(_):
    types.struct({"foo": types.i12})({"foo": 7})
    dim(types.i8, 2)([42, 45])
    types.i8(5)

    BarType({"foo": 7})

    Taps()
    StupidLegacy(ignore=no_connect)


@module
class ComplexPorts:
  clk = Input(types.i1)
  sel = Input(types.i2)
  data_in = Input(dim(32, 3))
  struct_data_in = Input(types.struct({"foo": types.i36}))

  a = Output(types.i32)
  b = Output(types.i32)
  c = Output(types.i32)

  @generator
  def build(ports):
    assert len(ports.data_in) == 3
    ports.set_all_ports({
        'a': ports.data_in[0].reg(ports.clk).reg(ports.clk),
        'b': ports.data_in[ports.sel],
        'c': ports.struct_data_in.foo[:-4]
    })


top = System([Top])
top.generate()
top.print()

# CHECK-LABEL:  msft.module @Top {} () attributes {fileName = "Top.sv"} {
# CHECK:    %c7_i12 = hw.constant 7 : i12
# CHECK:    hw.struct_create (%c7_i12) : !hw.struct<foo: i12>
# CHECK:    %c42_i8 = hw.constant 42 : i8
# CHECK:    %c45_i8 = hw.constant 45 : i8
# CHECK:    hw.array_create %c45_i8, %c42_i8 : i8
# CHECK:    %c5_i8 = hw.constant 5 : i8
# CHECK:    %c7_i12_0 = hw.constant 7 : i12
# CHECK:    hw.struct_create (%c7_i12_0) : !hw.typealias<@pycde::@bar, !hw.struct<foo: i12>>
# CHECK:    %Taps.taps = msft.instance @Taps @Taps()  : () -> !hw.array<3xi8>
# CHECK:    %c0_i4 = hw.constant 0 : i4
# CHECK:    [[ARG0:%.+]] = hw.bitcast %c0_i4 : (i4) -> !hw.array<4xi1>
# CHECK:    msft.instance @StupidLegacy @StupidLegacy([[ARG0]])  : (!hw.array<4xi1>) -> ()
# CHECK:    msft.output
# CHECK-LABEL:  msft.module @Taps {} () -> (taps: !hw.array<3xi8>) attributes {fileName = "Taps.sv"} {
# CHECK:    %c-53_i8 = hw.constant -53 : i8
# CHECK:    %c100_i8 = hw.constant 100 : i8
# CHECK:    %c23_i8 = hw.constant 23 : i8
# CHECK:    [[R0:%.+]] = hw.array_create %c23_i8, %c100_i8, %c-53_i8 : i8
# CHECK:    msft.output [[R0]] : !hw.array<3xi8>
# CHECK:  msft.module.extern @StupidLegacy(%ignore: !hw.array<4xi1>) attributes {verilogName = "StupidLegacy"}

sys = System([ComplexPorts])
sys.generate()
sys.print()
# CHECK:  msft.module @ComplexPorts {} (%clk: i1, %data_in: !hw.array<3xi32>, %sel: i2, %struct_data_in: !hw.struct<foo: i36>) -> (a: i32, b: i32, c: i32)
# CHECK:    %c0_i2 = hw.constant 0 : i2
# CHECK:    [[REG0:%.+]] = hw.array_get %data_in[%c0_i2] {sv.namehint = "data_in__0"} : !hw.array<3xi32>
# CHECK:    [[REGR1:%data_in__0__reg1]] = seq.compreg sym @data_in__0__reg1 [[REG0]], %clk : i32
# CHECK:    [[REGR2:%data_in__0__reg2]] = seq.compreg sym @data_in__0__reg2 [[REGR1]], %clk : i32
# CHECK:    [[REG1:%.+]] = hw.array_get %data_in[%sel] : !hw.array<3xi32>
# CHECK:    [[REG2:%.+]] = hw.struct_extract %struct_data_in["foo"] {sv.namehint = "struct_data_in__foo"} : !hw.struct<foo: i36>
# CHECK:    [[REG3:%.+]] = comb.extract [[REG2]] from 0 {sv.namehint = "struct_data_in__foo_0upto32"} : (i36) -> i32
# CHECK:    msft.output [[REGR2]], [[REG1]], [[REG3]] : i32, i32, i32
