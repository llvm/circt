# RUN: rm -rf %t
# RUN: %PYTHON% %s %t
# RUN: FileCheck %s --input-file %t/CompReg.sv

import pycde
from pycde import types, module, Input, Output

from pycde.dialects import seq
from pycde.module import generator

import sys


@module
class CompReg:
  clk = Input(types.i1)
  input = Input(types.i8)
  output = Output(types.i8)

  @generator
  def build(ports):
    compreg = seq.CompRegOp(types.i8, clk=ports.clk, input=ports.input)
    ports.output = compreg


mod = pycde.System([CompReg], name="CompReg", output_directory=sys.argv[1])
mod.print()
mod.generate()
mod.print()
mod.emit_outputs()

# CHECK: reg [7:0] [[NAME:.+]];
# CHECK: always_ff @(posedge clk)
# CHECK: [[NAME]] <= {{.+}}
