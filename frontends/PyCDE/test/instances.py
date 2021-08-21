# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import pycde
import circt.dialects.hw

from circt import msft


@pycde.module
class Nothing:

  @pycde.generator
  def construct(mod):
    return {}


@pycde.module
class UnParameterized:
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(mod):
    Nothing()
    return {"y": mod.x}


@pycde.module
class Test:
  inputs = []
  outputs = []

  @pycde.generator
  def build(_):
    c1 = circt.dialects.hw.ConstantOp.create(pycde.types.i1, 1)
    UnParameterized(x=c1)
    UnParameterized(x=c1)


# CHECK: hw.module @pycde.UnParameterized
# CHECK-NOT: hw.module @pycde.UnParameterized
t = pycde.System([Test])
t.generate(["construct"])
t.print()

mod = t.get_module("Test")
print(mod)

t.run_passes()

# CHECK-LABEL: === Hierarchy
print("=== Hierarchy")
# CHECK-NEXT: <instance: [pycde_UnParameterized]>
# CHECK-NEXT: <instance: [pycde_UnParameterized, pycde_Nothing]>
# CHECK-NEXT: <instance: [pycde_UnParameterized_0]>
# CHECK-NEXT: <instance: [pycde_UnParameterized_0, pycde_Nothing]>
t.walk_instances("Test", lambda inst: print(inst))


def place_inst(inst):
  global x, y
  inst.place("dsp_inst", msft.DSP, x, y, 0)
  x += 1
  y += 2


x = 0
y = 10
t.walk_instances("Test", place_inst)
t.print()

# CHECK-LABEL: === Tcl
print("=== Tcl")

# CHECK-LABEL: proc pycde_Test_config { parent }
# CHECK-NEXT:  set_location_assignment MPDSP_X0_Y10_N0 -to $parent|pycde_UnParameterized|dsp_inst
# CHECK-NEXT:  set_location_assignment MPDSP_X1_Y12_N0 -to $parent|pycde_UnParameterized|pycde_Nothing|dsp_inst
# CHECK-NEXT:  set_location_assignment MPDSP_X2_Y14_N0 -to $parent|pycde_UnParameterized_0|dsp_inst
# CHECK-NEXT:  set_location_assignment MPDSP_X3_Y16_N0 -to $parent|pycde_UnParameterized_0|pycde_Nothing|dsp_inst
t.print_tcl()
