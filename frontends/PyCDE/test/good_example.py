# RUN: rm -rf %t
# RUN: %PYTHON% %s %t

# This is intended to be a simple 'tutorial' example.  Run it as a test to
# ensure that we keep it up to date (ensure it doesn't crash).

from pycde import dim, module, generator, types, Clock, Input, Output
import pycde

import sys


@module
class Mux:
  clk = Clock()
  data = Input(dim(8, 14))
  sel = Input(types.i4)

  out = Output(types.i8)

  @generator
  def build(ports):
    sel_reg = ports.sel.reg()
    ports.out = ports.data.reg()[sel_reg].reg()


t = pycde.System([Mux], name="MuxDemo", output_directory=sys.argv[1])
t.compile()
