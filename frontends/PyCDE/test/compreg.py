# RUN: %PYTHON% %s | FileCheck %s

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
  def build(mod):
    compreg = seq.CompRegOp.create(types.i8, clk=mod.clk, input=mod.input)
    return {"output": compreg.data}


mod = pycde.System([CompReg])
mod.generate()
mod.print_verilog()

# CHECK: reg [7:0] [[NAME:.+]];
# CHECK: always @(posedge clk)
# CHECK: [[NAME]] <= {{.+}}
