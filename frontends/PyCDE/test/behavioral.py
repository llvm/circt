# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from re import A
from pycde import module, generator, types, Input, Output
from pycde.constructs import If, Else, Then
import pycde

import sys


@module
class IfDemo:
  cond = Input(types.i1)
  cond2 = Input(types.i1)
  a = Input(types.ui8)
  b = Input(types.ui8)

  out = Output(types.ui17)
  out2 = Output(types.ui24)

  @generator
  def build(ports):
    w = ports.a * ports.b
    with If(ports.cond):
      with Then:
        with If(ports.cond2):
          with Then:
            x = ports.a
          with Else:
            x = ports.b
        v = ports.a * x
        u = v * ports.b
      with Else:
        v = ports.b.as_uint(16)
        u = v * ports.a
    ports.out2 = u
    ports.out = v + w


# CHECK-LABEL: msft.module @IfDemo {} (%a: ui8, %b: ui8, %cond: i1, %cond2: i1) -> (out: ui17, out2: ui24)
# CHECK:         %0 = hwarith.mul %a, %b {sv.namehint = "a_mul_b"} : (ui8, ui8) -> ui16
# CHECK:         %1 = comb.mux %cond2, %a, %b {sv.namehint = "x"} : ui8
# CHECK:         %2 = hwarith.mul %a, %1 {sv.namehint = "v_thenvalue"} : (ui8, ui8) -> ui16
# CHECK:         %3 = hwarith.mul %2, %b {sv.namehint = "u_thenvalue"} : (ui16, ui8) -> ui24
# CHECK:         %4 = hwarith.cast %b {sv.namehint = "v_elsevalue"} : (ui8) -> ui16
# CHECK:         %5 = hwarith.mul %4, %a {sv.namehint = "u_elsevalue"} : (ui16, ui8) -> ui24
# CHECK:         %6 = comb.mux %cond, %2, %4 {sv.namehint = "v"} : ui16
# CHECK:         %7 = comb.mux %cond, %3, %5 {sv.namehint = "u"} : ui24
# CHECK:         %8 = hwarith.add %6, %0 {sv.namehint = "v_plus_a_mul_b"} : (ui16, ui16) -> ui17
# CHECK:         msft.output %8, %7 : ui17, ui24

t = pycde.System([IfDemo], name="BehavioralTest", output_directory=sys.argv[1])
t.generate()
t.print()
# t.emit_outputs()
