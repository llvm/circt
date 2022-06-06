# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import module, System, generator, dim, Input, Output, Value
import sys


def array_from_tuple(*input):
  return Value(input)


@module
class ComplexMux:

  Clk = Input(dim(1))
  In = Input(dim(3, 4, 5))
  Sel = Input(dim(1))
  Out = Output(dim(3, 4))
  OutArr = Output(dim(3, 4, 2))
  OutSlice = Output(dim(3, 4, 3))

  @generator
  def create(ports):
    clk = ports.Clk
    select_from = Value([ports.In[3].reg(clk).reg(clk, cycles=2), ports.In[1]])
    ports.Out = select_from[ports.Sel]

    ports.OutArr = array_from_tuple(ports.In[0], ports.In[1])
    ports.OutSlice = ports.In[0:3]


# CHECK-LABEL: msft.module @ComplexMux {} (%Clk: i1, %In: !hw.array<5xarray<4xi3>>, %Sel: i1) -> (Out: !hw.array<4xi3>, OutArr: !hw.array<2xarray<4xi3>>, OutSlice: !hw.array<3xarray<4xi3>>)
# CHECK:         %c3_i3 = hw.constant 3 : i3
# CHECK:         %0 = hw.array_get %In[%c3_i3] {sv.namehint = "In__3"} : !hw.array<5xarray<4xi3>>
# CHECK:         %In__3__reg1 = seq.compreg sym @In__3__reg1 %0, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg2 = seq.compreg sym @In__3__reg2 %In__3__reg1, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg3 = seq.compreg sym @In__3__reg3 %In__3__reg2, %Clk : !hw.array<4xi3>
# CHECK:         %c1_i3 = hw.constant 1 : i3
# CHECK:         [[R1:%.+]] = hw.array_get %In[%c1_i3] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R2:%.+]] = hw.array_create [[R1]], %In__3__reg3 : !hw.array<4xi3>
# CHECK:         [[R3:%.+]] = hw.array_get [[R2]][%Sel] : !hw.array<2xarray<4xi3>>
# CHECK:         %c0_i3 = hw.constant 0 : i3
# CHECK:         [[R4:%.+]] = hw.array_get %In[%c0_i3] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi3>>
# CHECK:         %c1_i3_0 = hw.constant 1 : i3
# CHECK:         [[R5:%.+]] = hw.array_get %In[%c1_i3_0] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R6:%.+]] = hw.array_create [[R5]], [[R4]] : !hw.array<4xi3>
# CHECK:         [[R7:%.+]] = hw.array_slice %In[%c0_i3_1] {sv.namehint = "In_0upto3"} : (!hw.array<5xarray<4xi3>>) -> !hw.array<3xarray<4xi3>>
# CHECK:         msft.output [[R3]], [[R6]], [[R7]] : !hw.array<4xi3>, !hw.array<2xarray<4xi3>>, !hw.array<3xarray<4xi3>>

s = System([ComplexMux], name="Muxing", output_directory=sys.argv[1])
s.generate()
s.print()

s.emit_outputs()
