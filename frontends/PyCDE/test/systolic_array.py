# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import dim, module, generator, types, Input, Output
from pycde.constructs import SystolicArray
import pycde

import pycde.dialects.comb as comb

import sys


@module
class Top:
  clk = Input(types.i1)
  row_data = Input(dim(8, 3))
  col_data = Input(dim(8, 2))
  out = Output(dim(8, 2, 3))

  @generator
  def build(mod):
    # If we just feed constants, CIRCT pre-computes the outputs in the
    # generated Verilog! Keep these for demo purposes.
    # row_data = dim(8, 3)([1, 2, 3]).value
    # col_data = dim(8, 2)([4, 5]).value

    # CHECK-LABEL: %{{.+}} = msft.systolic.array [%{{.+}} : 3 x i8] [%{{.+}} : 2 x i8] pe (%arg0, %arg1) -> (i8) {
    # CHECK:         [[SUM:%.+]] = comb.add %arg0, %arg1 : i8
    # CHECK:         [[SUMR:%.+]] = seq.compreg [[SUM]], %clk : i8
    # CHECK:         msft.pe.output [[SUMR]] : i8
    def pe(r, c):
      sum = comb.AddOp(r, c)
      return sum.reg(mod.clk)

    pe_outputs = SystolicArray(mod.row_data, mod.col_data, pe)

    mod.out = pe_outputs


t = pycde.System([Top], name="SATest", output_directory=sys.argv[1])
t.generate()
print("=== Pre-pass mlir dump")
t.print()

print("=== Running passes")
t.run_passes()

print("=== Final mlir dump")
t.print()

t.emit_outputs()
