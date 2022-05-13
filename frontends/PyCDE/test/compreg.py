# RUN: rm -rf %t
# RUN: %PYTHON% %s %t
# RUN: FileCheck %s --input-file %t/CompReg.sv
# RUN: FileCheck %s --input-file %t/CompReg.tcl --check-prefix TCL

import pycde
from pycde import types, module, Input, Output

from pycde.devicedb import PrimitiveType
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
    compreg = seq.CompRegOp(types.i8,
                            clk=ports.clk,
                            input=ports.input,
                            name="reg",
                            sym_name="reg")
    ports.output = compreg


mod = pycde.System([CompReg], name="CompReg", output_directory=sys.argv[1])
mod.print()
mod.generate()
top_inst = mod.get_instance(CompReg)
mod.createdb()
top_inst["reg"].place(PrimitiveType.FF, 0, 0, 0)
mod.print()
mod.emit_outputs()

# CHECK: reg [7:0] [[NAME:reg_.]];
# CHECK: always_ff @(posedge clk)
# CHECK: [[NAME]] <= {{.+}}

# TCL: set_location_assignment FF_X0_Y0_N0 -to $parent|reg_{{.}}
