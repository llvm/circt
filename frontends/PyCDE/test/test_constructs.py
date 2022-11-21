# RUN: %PYTHON% %s | FileCheck %s

from pycde import generator, types, dim
from pycde.common import Clock, Input, Output
from pycde.constructs import NamedWire, Reg, Wire, SystolicArray
from pycde.dialects import comb
from pycde.testing import unittestmodule

# CHECK-LABEL: msft.module @WireAndRegTest {} (%In: i8, %clk: i1) -> (Out: i8, OutReg: i8)
# CHECK:         %w1 = sv.wire  : !hw.inout<i8>
# CHECK:         [[r0:%.+]] = sv.read_inout %w1 : !hw.inout<i8>
# CHECK:         sv.assign %w1, %In : i8
# CHECK:         %in = sv.wire  : !hw.inout<i8>
# CHECK:         {{%.+}} = sv.read_inout %in : !hw.inout<i8>
# CHECK:         sv.assign %in, %In : i8
# CHECK:         [[r1:%.+]] = seq.compreg %In, %clk : i8
# CHECK:         msft.output [[r0]], [[r1]] : i8, i8


@unittestmodule()
class WireAndRegTest:
  In = Input(types.i8)
  clk = Clock()
  Out = Output(types.i8)
  OutReg = Output(types.i8)

  @generator
  def create(ports):
    w1 = Wire(types.i8, "w1")
    ports.Out = w1
    w1.assign(ports.In)

    NamedWire(ports.In, "in")

    r1 = Reg(types.i8)
    ports.OutReg = r1
    r1.assign(ports.In)


# CHECK-LABEL: %{{.+}} = msft.systolic.array [%{{.+}} : 3 x i8] [%{{.+}} : 2 x i8] pe (%arg0, %arg1) -> (i8) {
# CHECK:         [[SUM:%.+]] = comb.add %arg0, %arg1 {sv.namehint = "sum"} : i8
# CHECK:         [[SUMR:%.+]] = seq.compreg sym @sum__reg1 [[SUM]], %clk : i8
# CHECK:         msft.pe.output [[SUMR]] : i8


# CHECK-LABEL: hw.module @SystolicArrayTest<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%clk: i1, %col_data: !hw.array<2xi8>, %row_data: !hw.array<3xi8>) -> (out: !hw.array<3xarray<2xi8>>)
# CHECK:         %sum__reg1_0_0 = sv.reg sym @sum__reg1  : !hw.inout<i8>
# CHECK:         sv.read_inout %sum__reg1_0_0 : !hw.inout<i8>
@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class SystolicArrayTest:
  clk = Input(types.i1)
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
