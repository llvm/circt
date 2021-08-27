# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import pycde
import circt.dialects.hw

from circt import msft
from pycde.appid import AppIDIndex
import pycde.attributes as attrs


@pycde.externmodule
class Nothing:
  pass


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

mod = t.get_module("pycde.Test")
print(mod)

t.run_passes()

# CHECK-LABEL: === Hierarchy
print("=== Hierarchy")
# CHECK-NEXT: <instance: [pycde_UnParameterized]>
# CHECK-NEXT: <instance: [pycde_UnParameterized, pycde_Nothing]>
# CHECK-NEXT: <instance: [pycde_UnParameterized_0]>
# CHECK-NEXT: <instance: [pycde_UnParameterized_0, pycde_Nothing]>
t.walk_instances("pycde_Test", lambda inst: print(inst))


locs = pycde.AppIDIndex()
locs.lookup(pycde.AppID("pycde_UnParameterized_0"))["loc"] = \
  (["memory", "bank"], msft.M20K, 39, 25, 0)


def place_inst(inst):
  global x, y
  if inst.modname == "Nothing":
    inst.place("dsp_inst", msft.DSP, x, y)
    x += 1
    y += 2
  else:
    props = locs.lookup(inst.appid)
    if "loc" in props:
      inst.place(*props["loc"])


x = 0
y = 10
t.walk_instances("pycde_Test", place_inst)


instance_attrs = pycde.AppIDIndex()
loc = attrs.placement(["memory", "bank"], msft.M20K, 15, 25, 0)
instance_attrs.lookup(pycde.AppID("pycde_UnParameterized")).add_attribute(loc)
instance_attrs.lookup(pycde.AppID("pycde_UnParameterized",
                                  "pycde_Nothing")).add_attribute(loc)
t.walk_instances("pycde_Test", instance_attrs.apply_attributes_visitor)

assert instance_attrs.find_unused() is None
instance_attrs.lookup(pycde.AppID("doesnotexist")).add_attribute(loc)
assert (len(instance_attrs.find_unused()) == 1)

t.print()

# CHECK-LABEL: === Tcl
print("=== Tcl")

# CHECK-LABEL: proc pycde_Test_config { parent }
# CHECK-NEXT:  set_location_assignment MPDSP_X0_Y10_N0 -to $parent|pycde_UnParameterized|pycde_Nothing|dsp_inst
# CHECK-NEXT:  set_location_assignment M20K_X15_Y25_N0 -to $parent|pycde_UnParameterized|pycde_Nothing|memory|bank
# CHECK-NEXT:  set_location_assignment M20K_X15_Y25_N0 -to $parent|pycde_UnParameterized|memory|bank
# CHECK-NEXT:  set_location_assignment MPDSP_X1_Y12_N0 -to $parent|pycde_UnParameterized_0|pycde_Nothing|dsp_inst
# CHECK-NEXT:  set_location_assignment M20K_X39_Y25_N0 -to $parent|pycde_UnParameterized_0|memory|bank
t.print_tcl()
