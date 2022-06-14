# RUN: rm -rf %t
# RUN: %PYTHON% %s %t
# RUN: FileCheck %s --input-file %t/CompReg.sv
# RUN: FileCheck %s --input-file %t/CompReg.tcl --check-prefix TCL

import pycde
from pycde import types, module, Input, Output
from pycde.devicedb import LocationVector

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
    compreg = ports.input.reg(clk=ports.clk,
                              name="reg",
                              sv_attributes=["dont_merge"])
    ports.output = compreg


mod = pycde.System([CompReg], name="CompReg", output_directory=sys.argv[1])
mod.print()
mod.generate()
top_inst = mod.get_instance(CompReg)
mod.createdb()

locs = LocationVector(top_inst["reg"].type, [(0, 0, 0), None, (0, 0, 2),
                                             (0, 0, 3), (0, 0, 4), (0, 0, 5),
                                             (0, 0, 6), (0, 0, 7)])
print(locs)

top_inst["reg"].place([(0, 0, 0), None, (0, 0, 2), (0, 0, 3), (0, 0, 4),
                       (0, 0, 5), (0, 0, 6), (0, 0, 7)])
mod.print()
mod.emit_outputs()

# CHECK: (* dont_merge *)
# CHECK: reg [7:0] [[NAME:reg_.]];
# CHECK: always_ff @(posedge clk)
# CHECK: [[NAME]] <= {{.+}}

# TCL-DAG: set_location_assignment FF_X0_Y0_N7 -to $parent|reg_2[7]
# TCL-DAG: set_location_assignment FF_X0_Y0_N6 -to $parent|reg_2[6]
# TCL-DAG: set_location_assignment FF_X0_Y0_N5 -to $parent|reg_2[5]
# TCL-DAG: set_location_assignment FF_X0_Y0_N4 -to $parent|reg_2[4]
# TCL-DAG: set_location_assignment FF_X0_Y0_N3 -to $parent|reg_2[3]
# TCL-DAG: set_location_assignment FF_X0_Y0_N2 -to $parent|reg_2[2]
# TCL-DAG: set_location_assignment FF_X0_Y0_N0 -to $parent|reg_2[0]
# TCL-NOT: set_location_assignment FF_X0_Y0_N1 -to $parent|reg_2[1]
