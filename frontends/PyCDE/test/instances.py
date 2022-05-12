# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file %t/Test.tcl --check-prefix=OUTPUT

import pycde
import pycde.dialects.hw

from pycde.attributes import placement
from pycde.devicedb import (PhysLocation, PrimitiveDB, PrimitiveType)

import sys

from pycde.instance import InstanceDoesNotExistError, Instance


@pycde.externmodule
class Nothing:
  pass


@pycde.module
class UnParameterized:
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(mod):
    Nothing().name = "nothing_inst"
    mod.y = mod.x


@pycde.module
class Test:
  inputs = []
  outputs = []

  @pycde.generator
  def build(_):
    c1 = pycde.dialects.hw.ConstantOp(pycde.types.i1, 1)
    UnParameterized(x=c1).name = "unparam"
    UnParameterized(x=c1).name = "unparam"


# Set up the primitive locations. Errors out if location is placed but doesn't
# exist.
primdb = PrimitiveDB()
primdb.add_coords("M20K", 39, 25)
primdb.add_coords(PrimitiveType.M20K, 15, 25)
primdb.add_coords(PrimitiveType.M20K, 40, 40)
primdb.add_coords("DSP", 0, 10)
primdb.add_coords(PrimitiveType.DSP, 1, 12)
primdb.add(PhysLocation(PrimitiveType.DSP, 39, 25))

print(PhysLocation(PrimitiveType.DSP, 39, 25))
# CHECK: PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0>

# CHECK: msft.module @UnParameterized
# CHECK-NOT: msft.module @UnParameterized
t = pycde.System([Test], name="Test", output_directory=sys.argv[1])
t.generate(["construct"])
t.print()

Test.print()
UnParameterized.print()

# CHECK-LABEL: === Hierarchy
print("=== Hierarchy")
# CHECK-NEXT: <instance: []>
# CHECK-NEXT: <instance: [UnParameterized]>
# CHECK-NEXT: <instance: [UnParameterized, Nothing]>
# CHECK-NEXT: <instance: [UnParameterized_1]>
# CHECK-NEXT: <instance: [UnParameterized_1, Nothing]>
test_inst = t.get_instance(Test)
t.createdb(primdb)
test_inst.walk(lambda inst: print(inst))

# CHECK-LABEL: === Placements
print("=== Placements")


def place_inst(inst: Instance):
  if inst.name == "UnParameterized_1":
    inst.place(PrimitiveType.M20K, 39, 25, 0, "memory|bank")


t.get_instance(Test).walk(place_inst)

# TODO: Add back physical region support

# region1 = t.create_physical_region("region_0").add_bounds((0, 10), (0, 10))
# region1.add_bounds((10, 20), (10, 20))
# ref = region1.get_ref()
# instance_attrs.lookup(pycde.AppID("UnParameterized",
#                                   "Nothing")).add_attribute(ref)

# region_anon = t.create_physical_region()
# assert region_anon._physical_region.sym_name.value == "region_1"

# region_explicit = t.create_physical_region("region_1")
# assert region_explicit._physical_region.sym_name.value == "region_1_1"

test_inst = t.get_instance(Test)
t.createdb()

test_inst["UnParameterized"].attach_attribute(
    placement(["memory", "bank"], PrimitiveType.M20K, 15, 25, 0))
test_inst["UnParameterized"]["Nothing"].place(PrimitiveType.DSP, 39, 25, 0)

test_inst.walk(lambda inst: print(inst, inst.locations))
# CHECK: <instance: []> []
# CHECK: <instance: [UnParameterized]> [(PhysLocation<PrimitiveType.M20K, x:15, y:25, num:0>, 'memory|bank')]
# CHECK: <instance: [UnParameterized, Nothing]> [(PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0>, None)]
# CHECK: <instance: [UnParameterized_1]> [(PhysLocation<PrimitiveType.M20K, x:39, y:25, num:0>, 'memory|bank')]
# CHECK: <instance: [UnParameterized_1, Nothing]> []

# TODO: add back anonymous reservations

# reserved_loc = PhysLocation(PrimitiveType.M20K, 40, 40, 0)
# entity_extern = t.create_entity_extern("tag")
# test_inst.placedb.reserve_location(reserved_loc, entity_extern)

# CHECK: PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0> has ['UnParameterized', 'Nothing']
loc = PhysLocation(PrimitiveType.DSP, 39, 25, 0)
print(f"{loc} has {t.placedb.get_instance_at(loc)[0].path_names}")

assert t.placedb.get_instance_at(PhysLocation(PrimitiveType.M20K, 0, 0,
                                              0)) is None
# assert test_inst.placedb.get_instance_at(reserved_loc) is not None

instance_attrs = pycde.AppIDIndex()
assert instance_attrs.find_unused() is None
instance_attrs.lookup(pycde.AppID("doesnotexist")).add_attribute(loc)
assert (len(instance_attrs.find_unused()) == 1)

# CHECK-LABEL: === Force-clean all the caches and test rebuilds
print("=== Force-clean all the caches and test rebuilds")
t._op_cache.release_ops()

test_inst.walk(lambda inst: print(inst, inst.locations))
# CHECK: <instance: []> []
# CHECK: <instance: [UnParameterized]> [(PhysLocation<PrimitiveType.M20K, x:15, y:25, num:0>, 'memory|bank')]
# CHECK: <instance: [UnParameterized, Nothing]> [(PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0>, None)]
# CHECK: <instance: [UnParameterized_1]> [(PhysLocation<PrimitiveType.M20K, x:39, y:25, num:0>, 'memory|bank')]
# CHECK: <instance: [UnParameterized_1, Nothing]> []

# CHECK: PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0> has (<instance: [UnParameterized, Nothing]>, None)
print(f"{loc} has {t.placedb.get_instance_at(loc)}")

print("=== Pre-pass mlir dump")
t.print()

print("=== Running passes")
t.run_passes()

try:
  test_inst._dyn_inst()
  assert False
except InstanceDoesNotExistError:
  pass

print("=== Final mlir dump")
t.print()

# OUTPUT-LABEL: proc Test_config { parent }
# OUTPUT-NOT:  set_location_assignment M20K_X40_Y40
# OUTPUT-DAG:  set_location_assignment M20K_X39_Y25_N0 -to $parent|UnParameterized_1|memory|bank
# OUTPUT-DAG:  set_location_assignment M20K_X15_Y25_N0 -to $parent|UnParameterized|memory|bank
# OUTPUT-DAG:  set_location_assignment MPDSP_X39_Y25_N0 -to $parent|UnParameterized|Nothing
# OUTPUT-NOT:  set_location_assignment
# OUTPUT-NEXT: }
t.emit_outputs()
