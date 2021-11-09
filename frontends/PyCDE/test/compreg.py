# RUN: %PYTHON% %s
# RUN: FileCheck %s --input-file CompReg/CompReg.sv

import pycde
from pycde import types, module, Input, Output

from circt.dialects import seq
from pycde.module import generator


@module
class CompReg:
  clk = Input(types.i1)
  input = Input(types.i8)
  output = Output(types.i8)

  @generator
  def build(ports):
    compreg = seq.CompRegOp.create(types.i8, clk=ports.clk, input=ports.input)
    ports.output = compreg.data


mod = pycde.System([CompReg], name="CompReg")
mod.print()
mod.generate()
mod.print()
mod.emit_outputs()

# CHECK: reg [7:0] [[NAME:.+]];
# CHECK: always @(posedge clk)
# CHECK: [[NAME]] <= {{.+}}
