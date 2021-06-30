# RUN: %PYTHON% %s | FileCheck %s

import pycde
from pycde import types

from circt.dialects import seq


class CompReg(pycde.System):
  inputs = [("clk", types.i1), ("input", types.i8)]
  outputs = [("output", types.i8)]

  def build(self, top):
    compreg = seq.CompRegOp.create(types.i8, clk=top.clk, input=top.input)
    return {"output": compreg.data}


mod = CompReg()
mod.print_verilog()

# CHECK: reg [7:0] [[NAME:.+]];
# CHECK: always @(posedge clk)
# CHECK: [[NAME]] <= {{.+}}
