# RUN: rm -rf %t
# RUN: %PYTHON% %s %t

from pycde import module, generator, types, Input, Output
from pycde.constructs import If, Else, Then
import pycde

import sys


@module
class IfDemo:
  cond = Input(types.i1)
  out = Output(types.i8)

  @generator
  def build(ports):
    with If(ports.cond):
      with Then:
        v = types.i8(1)
      with Else:
        v = types.i8(0)
    ports.out = v


t = pycde.System([IfDemo], name="BehavioralTest", output_directory=sys.argv[1])
t.generate()
print("=== Pre-pass mlir dump")
t.print()

print("=== Running passes")
t.run_passes()

print("=== Final mlir dump")
t.print()

t.emit_outputs()
