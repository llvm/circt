# RUN: %PYTHON% %s | FileCheck %s

from pycde import (Output, Input, module, generator, types, dim, System)


@module
class WireNames:
  clk = Input(types.i1)
  sel = Input(types.i2)
  data_in = Input(dim(32, 3))

  a = Output(types.i32)
  b = Output(types.i32)

  @generator
  def build(mod):
    foo = mod.data_in[0]
    foo.name = "foo"
    arr_data = dim(32, 4).create([1, 2, 3, 4], "arr_data")
    return {
        'a': foo.reg(mod.clk).reg(mod.clk),
        'b': arr_data[mod.sel],
    }


sys = System([WireNames])
sys.generate()
sys.print()
# CHECK:  hw.module @pycde.WireNames(%clk: i1, %data_in: !hw.array<3xi32>, %sel: i2) -> (%a: i32, %b: i32) {
# CHECK:    %c0_i2 = hw.constant 0 : i2
# CHECK:    %0 = hw.array_get %data_in[%c0_i2] {name = "foo"} : !hw.array<3xi32>
# CHECK:    %c1_i32 = hw.constant 1 : i32
# CHECK:    %c2_i32 = hw.constant 2 : i32
# CHECK:    %c3_i32 = hw.constant 3 : i32
# CHECK:    %c4_i32 = hw.constant 4 : i32
# CHECK:    %1 = hw.array_create %c4_i32, %c3_i32, %c2_i32, %c1_i32 : i32
# CHECK:    %foo__reg1 = seq.compreg %0, %clk : i32
# CHECK:    %foo__reg2 = seq.compreg %foo__reg1, %clk : i32
# CHECK:    [[REG4:%.+]] = hw.array_get %1[%sel] : !hw.array<4xi32>
# CHECK:    hw.output %foo__reg2, [[REG4]] : i32, i32
# CHECK:  }

sys.print_verilog()
