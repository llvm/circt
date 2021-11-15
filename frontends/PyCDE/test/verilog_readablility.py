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
  def build(ports):
    foo = ports.data_in[0]
    foo.name = "foo"
    arr_data = dim(32, 4).create([1, 2, 3, 4], "arr_data")
    ports.set_all_ports({
        'a': foo.reg(ports.clk).reg(ports.clk),
        'b': arr_data[ports.sel],
    })


sys = System([WireNames])
sys.generate()
sys.run_passes()
sys.print()
# CHECK-LABEL:  hw.module @WireNames(%clk: i1, %data_in: !hw.array<3xi32>, %sel: i2) -> (a: i32, b: i32) {{.*}} {
# CHECK:    %c1_i32 = hw.constant 1 : i32
# CHECK:    %c2_i32 = hw.constant 2 : i32
# CHECK:    %c3_i32 = hw.constant 3 : i32
# CHECK:    %c4_i32 = hw.constant 4 : i32
# CHECK:    %{{.+}} = hw.array_create %c4_i32, %c3_i32, %c2_i32, %c1_i32 : i32
# CHECK:    %foo__reg1 = sv.reg  : !hw.inout<i32>
# CHECK:    %foo__reg2 = sv.reg  : !hw.inout<i32>
# CHECK:    %{{.+}} = sv.read_inout %foo__reg2 : !hw.inout<i32>
# CHECK:    sv.alwaysff(posedge %clk)  {
# CHECK:      %c0_i2 = hw.constant 0 : i2
# CHECK:      %{{.+}} = hw.array_get %data_in[%c0_i2] {name = "foo"} : !hw.array<3xi32>
# CHECK:      sv.passign %foo__reg1, %3 : i32
# CHECK:      %{{.+}} = sv.read_inout %foo__reg1 : !hw.inout<i32>
# CHECK:      sv.passign %foo__reg2, %4 : i32
# CHECK:    }
# CHECK:    %{{.+}} = hw.array_get %0[%sel] : !hw.array<4xi32>
# CHECK:    hw.output %1, %2 : i32, i32
# CHECK:  }
