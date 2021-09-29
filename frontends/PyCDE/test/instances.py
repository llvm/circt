# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import pycde
import circt.dialects.hw

from circt import msft
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


# CHECK: msft.module @UnParameterized
# CHECK-NOT: msft.module @UnParameterized
t = pycde.System([Test])
t.generate(["construct"])
t.print()

print(Test)
Test.print()

t.run_passes()
Test.print()

# CHECK-LABEL: === Hierarchy
print("=== Hierarchy")
# CHECK-NEXT: <instance: [UnParameterized]>
# CHECK-NEXT: <instance: [UnParameterized, Nothing]>
# CHECK-NEXT: <instance: [UnParameterized_0]>
# CHECK-NEXT: <instance: [UnParameterized_0, Nothing]>
mod = t.get_instance(Test).walk(lambda inst: print(inst))

locs = pycde.AppIDIndex()
locs.lookup(pycde.AppID("UnParameterized_0"))["loc"] = \
  (["memory", "bank"], msft.M20K, 39, 25, 0)


def place_inst(inst):
  global x, y
  if inst.module == Nothing:
    inst.place("dsp_inst", msft.DSP, x, y)
    x += 1
    y += 2
  else:
    props = locs.lookup(inst.appid)
    if "loc" in props:
      inst.place(*props["loc"])


x = 0
y = 10
t.get_instance(Test).walk(place_inst)

instance_attrs = pycde.AppIDIndex()
loc = attrs.placement(["memory", "bank"], msft.M20K, 15, 25, 0)
instance_attrs.lookup(pycde.AppID("UnParameterized")).add_attribute(loc)
loc = attrs.placement(["memory", "bank"], msft.DSP, 39, 25, 0)
instance_attrs.lookup(pycde.AppID("UnParameterized",
                                  "Nothing")).add_attribute(loc)
test_inst = t.get_instance(Test)
test_inst.walk(instance_attrs.apply_attributes_visitor)

assert test_inst.get_instance_at(loc) is not None
assert test_inst.get_instance_at(msft.PhysLocationAttr.get(msft.M20K, 0, 0,
                                                           0)) is None

assert instance_attrs.find_unused() is None
instance_attrs.lookup(pycde.AppID("doesnotexist")).add_attribute(loc)
assert (len(instance_attrs.find_unused()) == 1)

print("=== Final mlir dump")
t.print()

# CHECK-LABEL: === Tcl
print("=== Tcl")

# CHECK-LABEL: proc Test_config { parent }
# CHECK-DAG:  set_location_assignment MPDSP_X0_Y10_N0 -to $parent|UnParameterized|Nothing|dsp_inst
# CHECK-DAG:  set_location_assignment MPDSP_X39_Y25_N0 -to $parent|UnParameterized|Nothing|memory|bank
# CHECK-DAG:  set_location_assignment M20K_X15_Y25_N0 -to $parent|UnParameterized|memory|bank
# CHECK-DAG:  set_location_assignment MPDSP_X1_Y12_N0 -to $parent|UnParameterized_0|Nothing|dsp_inst
# CHECK-DAG:  set_location_assignment M20K_X39_Y25_N0 -to $parent|UnParameterized_0|memory|bank
t.print_tcl(Test)
