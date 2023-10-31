# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import om
from circt.ir import Context, InsertionPoint, Location, Module
from circt.support import var_to_attribute

from dataclasses import dataclass

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  module = Module.parse("""
  module {
    om.class @node() {
      %0 = om.constant #om.list<!om.string,["MyThing" : !om.string]> : !om.list<!om.string>
      %1 = om.constant "Component.inst1.foo" : !om.string
      om.class.field @field2, %1 : !om.string
    }
    
    om.class @comp(
        %inst1_propOut_bore: !om.class.type<@node>,
        %inst2_propOut_bore: !om.class.type<@node>) {
      om.class.field @field2, %inst1_propOut_bore : !om.class.type<@node>
      om.class.field @field3, %inst2_propOut_bore : !om.class.type<@node>
    }
    
    om.class  @Client() {
      %0 = om.object @node() : () -> !om.class.type<@node>
      %2 = om.object @comp(%0, %0) : (!om.class.type<@node>, !om.class.type<@node>) -> !om.class.type<@comp>
    
      om.class.field @client_omnode_0_OMIROut, %2 : !om.class.type<@comp>
      om.class.field @node0_OMIROut, %0 : !om.class.type<@node>
      om.class.field @node1_OMIROut, %0 : !om.class.type<@node>
    }

    %sym = om.constant #om.ref<<@Root::@x>> : !om.ref

    om.class @Test(%param: !om.integer) {
      om.class.field @field, %param : !om.integer

      %c_14 = om.constant #om.integer<14> : !om.integer
      %0 = om.object @Child(%c_14) : (!om.integer) -> !om.class.type<@Child>
      om.class.field @child, %0 : !om.class.type<@Child>

      om.class.field @reference, %sym : !om.ref

      %list = om.constant #om.list<!om.string, ["X" : !om.string, "Y" : !om.string]> : !om.list<!om.string>
      om.class.field @list, %list : !om.list<!om.string>

      %tuple = om.tuple_create %list, %c_14: !om.list<!om.string>, !om.integer
      om.class.field @tuple, %tuple : tuple<!om.list<!om.string>, !om.integer>

      %c_15 = om.constant #om.integer<15> : !om.integer
      %1 = om.object @Child(%c_15) : (!om.integer) -> !om.class.type<@Child>
      %list_child = om.list_create %0, %1: !om.class.type<@Child>
      %2 = om.object @Nest(%list_child) : (!om.list<!om.class.type<@Child>>) -> !om.class.type<@Nest>
      om.class.field @nest, %2 : !om.class.type<@Nest>

      %3 = om.constant #om.map<!om.integer, {a = #om.integer<42>, b = #om.integer<32>}> : !om.map<!om.string, !om.integer>
      om.class.field @map, %3 : !om.map<!om.string, !om.integer>

      %x = om.constant "X" : !om.string
      %y = om.constant "Y" : !om.string
      %entry1 = om.tuple_create %x, %c_14: !om.string, !om.integer
      %entry2 = om.tuple_create %y, %c_15: !om.string, !om.integer

      %map = om.map_create %entry1, %entry2: !om.string, !om.integer
      om.class.field @map_create, %map : !om.map<!om.string, !om.integer>
    }

    om.class @Child(%0: !om.integer) {
      om.class.field @foo, %0 : !om.integer
    }

    om.class @Nest(%0: !om.list<!om.class.type<@Child>>) {
      om.class.field @list_child, %0 : !om.list<!om.class.type<@Child>>
    }

    hw.module @Root(in %clock: i1) {
      %0 = sv.wire sym @x : !hw.inout<i1>
    }
    
    om.class @Paths(%basepath: !om.frozenbasepath) {
      %0 = om.frozenbasepath_create %basepath "Foo/bar"
      %1 = om.frozenpath_create reference %0 "Bar/baz:Baz>w"
      om.class.field @path, %1 : !om.frozenpath

      %3 = om.frozenpath_empty
      om.class.field @deleted, %3 : !om.frozenpath
    }
  }
  """)

  evaluator = om.Evaluator(module)

# Test instantiate failure.

try:
  obj = evaluator.instantiate("Test")
except ValueError as e:
  # CHECK: actual parameter list length (0) does not match
  # CHECK: actual parameters:
  # CHECK: formal parameters:
  # CHECK: unable to instantiate object, see previous error(s)
  print(e)

# Test get field failure.

try:
  obj = evaluator.instantiate("Test", 42)
  obj.foo
except ValueError as e:
  # CHECK: field "foo" does not exist
  # CHECK: see current operation:
  # CHECK: unable to get field, see previous error(s)
  print(e)

# Test instantiate success.

obj = evaluator.instantiate("Test", 42)

assert isinstance(obj.type, om.ClassType)

# CHECK: Test
print(obj.type.name)

# CHECK: 42
print(obj.field)

# location of the om.class.field @field
# CHECK: loc("-":28:7)
print(obj.get_field_loc("field"))

# CHECK: 14
print(obj.child.foo)
# CHECK: loc("-":61:7)
print(obj.child.get_field_loc("foo"))
# CHECK: ('Root', 'x')
print(obj.reference)
(fst, snd) = obj.tuple
# CHECK: 14
print(snd)

# CHECK: loc("-":40:7)
print(obj.get_field_loc("tuple"))

try:
  print(obj.tuple[3])
except IndexError as e:
  # CHECK: tuple index out of range
  print(e)

for (name, field) in obj:
  # location from om.class.field @child, %0 : !om.class.type<@Child>
  # CHECK: name: child, field: <circt.dialects.om.Object object
  # CHECK-SAME: loc: loc("-":32:7)
  # location from om.class.field @field, %param : !om.integer
  # CHECK: name: field, field: 42
  # CHECK-SAME: loc: loc("-":28:7)
  # location from om.class.field @reference, %sym : !om.ref
  # CHECK: name: reference, field: ('Root', 'x')
  # CHECK-SAME: loc: loc("-":34:7)
  loc = obj.get_field_loc(name)
  print(f"name: {name}, field: {field}, loc: {loc}")

# CHECK: ['X', 'Y']
print(obj.list)
for child in obj.nest.list_child:
  # CHECK: 14
  # CHECK-NEXT: 15
  print(child.foo)

# CHECK: 2
print(len(obj.map))
# CHECK: {'a': 42, 'b': 32}
print(obj.map)
for k, v in obj.map.items():
  # CHECK-NEXT: a 42
  # CHECK-NEXT: b 32
  print(k, v)

try:
  print(obj.map_create[1])
except KeyError as e:
  # CHECK-NEXT: 'key is not integer'
  print(e)
try:
  print(obj.map_create["INVALID"])
except KeyError as e:
  # CHECK-NEXT: 'key not found'
  print(e)
# CHECK-NEXT: 14
print(obj.map_create["X"])

for k, v in obj.map_create.items():
  # CHECK-NEXT: X 14
  # CHECK-NEXT: Y 15
  print(k, v)

obj = evaluator.instantiate("Client")
object_dict: dict[om.Object, str] = {}
for field_name, data in obj:
  if isinstance(data, om.Object):
    object_dict[data] = field_name
assert len(object_dict) == 2

obj = evaluator.instantiate("Test", 41)
# CHECK: 41
print(obj.field)

path = om.BasePath.get_empty(evaluator.module.context)
obj = evaluator.instantiate("Paths", path)
print(obj.path)
# CHECK: OMReferenceTarget:~Foo|Foo/bar:Bar/baz:Baz>w

print(obj.deleted)
# CHECK: OMDeleted

paths_class = [
    cls for cls in module.body
    if hasattr(cls, "sym_name") and cls.sym_name.value == "Paths"
][0]
base_path_type = paths_class.regions[0].blocks[0].arguments[0].type
assert isinstance(base_path_type, om.BasePathType)

paths_fields = [
    op for op in paths_class.regions[0].blocks[0]
    if isinstance(op, om.ClassFieldOp)
]
for paths_field in paths_fields:
  assert isinstance(paths_field.value.type, om.PathType)
