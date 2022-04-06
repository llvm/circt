# RUN: rm -rf %t
# RUN: %PYTHON% %s %t

# This is intended to be a simple 'tutorial' example.  Run it as a test to
# ensure that we keep it up to date (ensure it doesn't crash).

from pycde import dim, module, generator, types, Input, Output
import pycde

import sys


@module
class Mux:
  clk = Input(types.i1)
  data = Input(dim(8, 14))
  sel = Input(types.i4)

  out = Output(types.i8)

  @generator
  def build(ports):
    sel_reg = ports.sel.reg(ports.clk)
    ports.out = ports.data.reg(ports.clk)[sel_reg].reg(ports.clk)


t = pycde.System([Mux], name="MuxDemo", output_directory=sys.argv[1])
t.generate()
t.emit_outputs()
