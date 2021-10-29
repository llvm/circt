# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import pycde
import circt.dialects.hw

from pycde.attributes import placement, logic_locked_region
from pycde.devicedb import PhysLocation, PrimitiveType


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
    mod.y = mod.x


@pycde.module
class Test:
  inputs = []
  outputs = []

  @pycde.generator
  def build(_):
    c1 = circt.dialects.hw.ConstantOp.create(pycde.types.i1, 1)
    UnParameterized(x=c1)
    UnParameterized(x=c1)


# Set up the primitive locations. Errors out if location is placed but doesn't
# exist.
primdb = pycde.PrimitiveDB()
primdb.add_coords("M20K", 39, 25)
primdb.add_coords(PrimitiveType.M20K, 15, 25)
primdb.add_coords("DSP", 0, 10)
primdb.add_coords(PrimitiveType.DSP, 1, 12)
primdb.add(PhysLocation(PrimitiveType.DSP, 39, 25))

print(PhysLocation(PrimitiveType.DSP, 39, 25))
# CHECK: PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0>

# CHECK: msft.module @UnParameterized
# CHECK-NOT: msft.module @UnParameterized
t = pycde.System([Test], primdb)
t.generate(["construct"])
t.print()

Test.print()
UnParameterized.print()

t.run_passes()

# CHECK-LABEL: === Hierarchy
print("=== Hierarchy")
# CHECK-NEXT: <instance: [UnParameterized]>
# CHECK-NEXT: <instance: [UnParameterized, Nothing]>
# CHECK-NEXT: <instance: [UnParameterized_1]>
# CHECK-NEXT: <instance: [UnParameterized_1, Nothing]>
mod = t.get_instance(Test).walk(lambda inst: print(inst))

locs = pycde.AppIDIndex()
locs.lookup(pycde.AppID("UnParameterized_1"))["loc"] = \
  (["memory", "bank"], PrimitiveType.M20K, 39, 25, 0)


def place_inst(inst):
  global x, y
  if inst.module == Nothing:
    inst.place("dsp_inst", PrimitiveType.DSP, x, y)
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
loc = placement(["memory", "bank"], PrimitiveType.M20K, 15, 25, 0)
instance_attrs.lookup(pycde.AppID("UnParameterized")).add_attribute(loc)
loc = placement(["memory", "bank"], PrimitiveType.DSP, 39, 25, 0)
instance_attrs.lookup(pycde.AppID("UnParameterized",
                                  "Nothing")).add_attribute(loc)
region = logic_locked_region(["reg1"], "region1", 0, 10, 0, 10)
instance_attrs.lookup(pycde.AppID("UnParameterized")).add_attribute(region)
test_inst = t.get_instance(Test)
test_inst.walk(instance_attrs.apply_attributes_visitor)

assert test_inst.placedb.get_instance_at(loc[1]) is not None
assert test_inst.placedb.get_instance_at(
    PhysLocation(PrimitiveType.M20K, 0, 0, 0)) is None

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
# CHECK-DAG:  set_location_assignment MPDSP_X1_Y12_N0 -to $parent|UnParameterized_1|Nothing|dsp_inst
# CHECK-DAG:  set_location_assignment M20K_X39_Y25_N0 -to $parent|UnParameterized_1|memory|bank
# CHECK-DAG:  set_instance_assignment -name PLACE_REGION "X0 Y0 X10 Y10" -to $parent|UnParameterized|reg1
# CHECK-DAG:  set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|UnParameterized|reg1
# CHECK-DAG:  set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|UnParameterized|reg1
# CHECK-DAG:  set_instance_assignment -name REGION_NAME region1 -to $parent|UnParameterized|reg1
t.print_tcl(Test)
