# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, dim, Clock, Input, Output, Module
from pycde.signals import Signal
from pycde.constructs import Mux
from pycde.testing import unittestmodule
from pycde.types import Bit, Bits

# CHECK-LABEL: hw.module @ComplexMux(in %Clk : !seq.clock, in %In : !hw.array<5xarray<4xi3>>, in %Sel : i1, out Out : !hw.array<4xi3>, out OutArr : !hw.array<2xarray<4xi3>>, out OutInt : i1, out OutSlice : !hw.array<3xarray<4xi3>>)
# CHECK:         %c3_i3 = hw.constant 3 : i3
# CHECK:         %0 = hw.array_get %In[%c3_i3] {sv.namehint = "In__3"} : !hw.array<5xarray<4xi3>>
# CHECK:         %In__3__reg1 = seq.compreg sym @In__3__reg1 %0, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg2 = seq.compreg sym @In__3__reg2 %In__3__reg1, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg3 = seq.compreg sym @In__3__reg3 %In__3__reg2, %Clk : !hw.array<4xi3>
# CHECK:         %c1_i3 = hw.constant 1 : i3
# CHECK:         [[R1:%.+]] = hw.array_get %In[%c1_i3] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R3:%.+]] = comb.mux bin %Sel, [[R1]], %In__3__reg3 {sv.namehint = "mux_Sel_In__3__reg3_In__1"} : !hw.array<4xi3>
# CHECK:         %c0_i3 = hw.constant 0 : i3
# CHECK:         [[R4:%.+]] = hw.array_get %In[%c0_i3] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi3>>
# CHECK:         %c1_i3_0 = hw.constant 1 : i3
# CHECK:         [[R5:%.+]] = hw.array_get %In[%c1_i3_0] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R6:%.+]] = hw.array_create [[R5]], [[R4]] : !hw.array<4xi3>
# CHECK:         [[R7:%.+]] = hw.array_slice %In[%c0_i3_1] {sv.namehint = "In_0upto3"} : (!hw.array<5xarray<4xi3>>) -> !hw.array<3xarray<4xi3>>
# CHECK:         %c0_i3_2 = hw.constant 0 : i3
# CHECK:         [[R8:%.+]] = hw.array_get %In[%c0_i3_2] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi3>>, i3
# CHECK:         [[R9:%.+]] = hw.array_get [[R8]][%c0_i2] {sv.namehint = "In__0__0"} : !hw.array<4xi3>
# CHECK:         %false = hw.constant false
# CHECK:         [[RN9:%.+]] = comb.concat %false, %Sel {sv.namehint = "Sel_padto_2"} : i1, i1
# CHECK:         %false_3 = hw.constant false
# CHECK:         [[R10:%.+]] = comb.concat %false_3, [[RN9]] {sv.namehint = "Sel_padto_2_padto_3"} : i1, i2
# CHECK:         [[R11:%.+]] = comb.shru bin [[R9]], [[R10]] : i3
# CHECK:         [[R12:%.+]] = comb.extract [[R11]] from 0 : (i3) -> i1
# CHECK:         hw.output [[R3]], [[R6]], [[R12]], [[R7]] : !hw.array<4xi3>, !hw.array<2xarray<4xi3>>, i1, !hw.array<3xarray<4xi3>>


@unittestmodule()
class ComplexMux(Module):

  Clk = Clock()
  In = Input(dim(3, 4, 5))
  Sel = Input(dim(1))
  Out = Output(dim(3, 4))
  OutArr = Output(dim(3, 4, 2))
  OutInt = Output(Bit)
  OutSlice = Output(dim(3, 4, 3))

  @generator
  def create(ports):
    ports.Out = Mux(ports.Sel, ports.In[3].reg().reg(cycles=2), ports.In[1])

    ports.OutArr = Signal.create([ports.In[0], ports.In[1]])
    ports.OutSlice = ports.In[0:3]

    ports.OutInt = ports.In[0][0][ports.Sel.pad_or_truncate(2)]


# -----


# CHECK-LABEL:  hw.module @SimpleMux2(in %op : i1, in %a : i32, in %b : i32, out out : i32)
# CHECK-NEXT:     [[r0:%.+]] = comb.mux bin %op, %b, %a
# CHECK-NEXT:     hw.output %0 : i32
@unittestmodule()
class SimpleMux2(Module):
  op = Input(Bits(1))
  a = Input(Bits(32))
  b = Input(Bits(32))
  out = Output(Bits(32))

  @generator
  def construct(self):
    self.out = Mux(self.op, self.a, self.b)


# CHECK-LABEL:  hw.module @SimpleMux4(in %op : i2, in %a : i32, in %b : i32, in %c : i32, in %d : i32, out out : i32)
# CHECK-NEXT:     [[r0:%.+]] = hw.array_create %d, %c, %b, %a
# CHECK-NEXT:     [[r1:%.+]] = hw.array_get [[r0]][%op]
# CHECK-NEXT:     hw.output [[r1]] : i32
@unittestmodule()
class SimpleMux4(Module):
  op = Input(Bits(2))
  a = Input(Bits(32))
  b = Input(Bits(32))
  c = Input(Bits(32))
  d = Input(Bits(32))
  out = Output(Bits(32))

  @generator
  def construct(self):
    self.out = Mux(self.op, self.a, self.b, self.c, self.d)
