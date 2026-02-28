# RUN: %PYTHON% %s | FileCheck %s

from pycde import generator, dim, Module
from pycde.common import Clock, Input, Output
from pycde.constructs import ControlReg, NamedWire, Reg, Wire, SystolicArray
from pycde.dialects import comb
from pycde.testing import unittestmodule
from pycde.types import Bit, Bits

I1 = Bit
I8 = Bits(8)

# CHECK-LABEL: hw.module @WireAndRegTest(in %In : i8, in %InCE : i1, in %clk : !seq.clock, in %rst : i1, out Out : i8, out OutReg : i8, out OutRegRst : i8, out OutRegCE : i8)
# CHECK:         [[r0:%.+]] = comb.extract %In from 0 {sv.namehint = "In_0upto7"} : (i8) -> i7
# CHECK:         [[r1:%.+]] = comb.extract %In from 7 {sv.namehint = "In_7upto8"} : (i8) -> i1
# CHECK:         [[r2:%.+]] = comb.concat [[r1]], [[r0]] {sv.namehint = "w1"} : i1, i7
# CHECK:         %in = sv.wire sym @in : !hw.inout<i8>
# CHECK:         {{%.+}} = sv.read_inout %in {sv.namehint = "in"} : !hw.inout<i8>
# CHECK:         sv.assign %in, %In : i8
# CHECK:         [[r1:%.+]] = seq.compreg %In, %clk : i8
# CHECK:         %c0_i8{{.*}} = hw.constant 0 : i8
# CHECK:         [[r5:%.+]] = seq.compreg %In, %clk reset %rst, %c0_i8{{.*}}  : i8
# CHECK:         [[r6:%.+]] = seq.compreg.ce %In, %clk, %InCE : i8
# CHECK:         hw.output [[r2]], [[r1]], [[r5]], [[r6]] : i8, i8, i8, i8


@unittestmodule()
class WireAndRegTest(Module):
  In = Input(I8)
  InCE = Input(I1)
  clk = Clock()
  rst = Input(I1)
  Out = Output(I8)
  OutReg = Output(I8)
  OutRegRst = Output(I8)
  OutRegCE = Output(I8)

  @generator
  def create(ports):
    w1 = Wire(I8, "w1")
    ports.Out = w1
    w1[0:7] = ports.In[0:7]
    w1[7] = ports.In[7]

    NamedWire(ports.In, "in")

    r1 = Reg(I8)
    ports.OutReg = r1
    r1.assign(ports.In)

    r_rst = Reg(I8, rst=ports.rst, rst_value=0)
    ports.OutRegRst = r_rst
    r_rst.assign(ports.In)

    r_ce = Reg(I8, ce=ports.InCE)
    ports.OutRegCE = r_ce
    r_ce.assign(ports.In)


# CHECK-LABEL: %{{.+}} = msft.systolic.array [%{{.+}} : 3 x i8] [%{{.+}} : 2 x i8] pe (%arg0, %arg1) -> (i8) {
# CHECK:         [[SUM:%.+]] = comb.add bin %arg0, %arg1 {sv.namehint = "sum"} : i8
# CHECK:         [[SUMR:%.+]] = seq.compreg sym @sum__reg1 [[SUM]], %clk : i8
# CHECK:         msft.pe.output [[SUMR]] : i8


# CHECK-LABEL: hw.module @SystolicArrayTest(in %clk : i1, in %col_data : !hw.array<2xi8>, in %row_data : !hw.array<3xi8>, out out : !hw.array<3xarray<2xi8>>)
# CHECK:         %sum__reg1_0_0 = sv.reg sym @sum__reg1  : !hw.inout<i8>
# CHECK:         sv.read_inout %sum__reg1_0_0 : !hw.inout<i8>
@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class SystolicArrayTest(Module):
  clk = Clock()
  col_data = Input(dim(8, 2))
  row_data = Input(dim(8, 3))
  out = Output(dim(8, 2, 3))

  @generator
  def build(ports):
    # If we just feed constants, CIRCT pre-computes the outputs in the
    # generated Verilog! Keep these for demo purposes.
    # row_data = dim(8, 3)([1, 2, 3])
    # col_data = dim(8, 2)([4, 5])

    def pe(r, c):
      sum = comb.AddOp(r, c)
      sum.name = "sum"
      return sum.reg(ports.clk)

    pe_outputs = SystolicArray(ports.row_data, ports.col_data, pe)

    ports.out = pe_outputs


# CHECK-LABEL:  hw.module @ControlReg_num_asserts2_num_resets1
# CHECK:          [[r0:%.+]] = hw.array_get %asserts[%false]
# CHECK:          [[r1:%.+]] = hw.array_get %asserts[%true]
# CHECK:          [[r2:%.+]] = comb.or bin [[r0]], [[r1]]
# CHECK:          [[r3:%.+]] = hw.array_get %resets[%c0_i0]
# CHECK:          [[r4:%.+]] = comb.or bin [[r3]]
# CHECK:          %state = seq.compreg [[r6]], %clk reset %rst, %false{{.*}}
# CHECK:          [[r5:%.+]] = comb.mux bin [[r4]], %false{{.*}}, %state
# CHECK:          [[r6:%.+]] = comb.mux bin [[r2]], %true{{.*}}, [[r5]]
# CHECK:          hw.output %state
@unittestmodule()
class ControlRegTest(Module):
  clk = Clock()
  rst = Input(I1)
  a1 = Input(I1)
  a2 = Input(I1)
  r1 = Input(I1)

  @generator
  def build(ports):
    ControlReg(ports.clk,
               ports.rst,
               asserts=[ports.a1, ports.a2],
               resets=[ports.r1])
