# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import om
from circt.ir import Context, InsertionPoint, Location, Module, IntegerAttr, IntegerType, Type
from circt.support import var_to_attribute

from dataclasses import dataclass
from typing import Dict

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  module = Module.parse("""
  module {
    om.class @node() -> (field2: !om.string) {
      %0 = om.constant #om.list<!om.string,["MyThing" : !om.string]> : !om.list<!om.string>
      %1 = om.constant "Component.inst1.foo" : !om.string
      om.class.fields %1 : !om.string
    }
    
    om.class @comp(
        %inst1_propOut_bore: !om.class.type<@node>,
        %inst2_propOut_bore: !om.class.type<@node>) -> (field2: !om.class.type<@node>, field3: !om.class.type<@node>) {
      om.class.fields %inst1_propOut_bore, %inst2_propOut_bore : !om.class.type<@node>, !om.class.type<@node>
    }
    
    om.class  @Client() -> (client_omnode_0_OMIROut: !om.class.type<@comp>, node0_OMIROut : !om.class.type<@node>, node1_OMIROut : !om.class.type<@node>) {
      %0 = om.object @node() : () -> !om.class.type<@node>
      %2 = om.object @comp(%0, %0) : (!om.class.type<@node>, !om.class.type<@node>) -> !om.class.type<@comp>
    
      om.class.fields %2, %0, %0 : !om.class.type<@comp>, !om.class.type<@node>, !om.class.type<@node>
    }

    om.class @Test(%param: !om.integer) -> (field: !om.integer, child: !om.class.type<@Child>, reference: !om.ref, list: !om.list<!om.string>, nest: !om.class.type<@Nest>, true: i1, false: i1) {
      %sym = om.constant #om.ref<<@Root::@x>> : !om.ref

      %c_14 = om.constant #om.integer<14 : si32> : !om.integer
      %0 = om.object @Child(%c_14) : (!om.integer) -> !om.class.type<@Child>


      %list = om.constant #om.list<!om.string, ["X" : !om.string, "Y" : !om.string]> : !om.list<!om.string>

      %c_15 = om.constant #om.integer<15 : si32> : !om.integer
      %1 = om.object @Child(%c_15) : (!om.integer) -> !om.class.type<@Child>
      %list_child = om.list_create %0, %1: !om.class.type<@Child>
      %2 = om.object @Nest(%list_child) : (!om.list<!om.class.type<@Child>>) -> !om.class.type<@Nest>




      %true = om.constant true
      %false = om.constant false

      om.class.fields %param, %0, %sym, %list, %2, %true, %false : !om.integer, !om.class.type<@Child>, !om.ref, !om.list<!om.string>, !om.class.type<@Nest>, i1, i1
    }

    om.class @Child(%0: !om.integer) -> (foo: !om.integer) {
      om.class.fields %0 : !om.integer
    }

    om.class @Nest(%0: !om.list<!om.class.type<@Child>>) -> (list_child: !om.list<!om.class.type<@Child>>) {
      om.class.fields %0 : !om.list<!om.class.type<@Child>>
    }

    hw.module @Root(in %clock: i1) {
      %0 = sv.wire sym @x : !hw.inout<i1>
    }
    
    om.class @Paths(%basepath: !om.frozenbasepath) -> (path: !om.frozenpath, deleted: !om.frozenpath) {
      %0 = om.frozenbasepath_create %basepath "Foo/bar"
      %1 = om.frozenpath_create reference %0 "Bar/baz:Baz>w"

      %3 = om.frozenpath_empty
      om.class.fields %1, %3 : !om.frozenpath, !om.frozenpath
    }

    om.class @Class1(%input: !om.integer) -> (value: !om.integer, input: !om.integer) {
      %0 = om.constant #om.integer<1 : si3> : !om.integer
      om.class.fields %0, %input : !om.integer, !om.integer
    }

    om.class @Class2() -> (value: !om.integer) {
      %0 = om.constant #om.integer<2 : si3> : !om.integer
      om.class.fields %0 : !om.integer
    }

    om.class @IntegerBinaryArithmeticObjectsDelayed() -> (result: !om.integer) {
      %0 = om.object @Class1(%5) : (!om.integer) -> !om.class.type<@Class1>
      %1 = om.object.field %0, [@value] : (!om.class.type<@Class1>) -> !om.integer

      %2 = om.object @Class2() : () -> !om.class.type<@Class2>
      %3 = om.object.field %2, [@value] : (!om.class.type<@Class2>) -> !om.integer

      %5 = om.integer.add %1, %3 : !om.integer
      om.class.fields %5 : !om.integer
    }
    om.class @AppendList(%head: !om.string, %tail: !om.list<!om.string>) -> (result: !om.list<!om.string>) {
      %0 = om.list_create %head : !om.string
      %1 = om.list_concat %0, %tail : !om.list<!om.string>
      om.class.fields %1 : !om.list<!om.string>
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
# CHECK: field: loc("-":{{.*}}:{{.*}})
print("field:", obj.get_field_loc("field"))

# CHECK: child.foo: 14
print("child.foo: ", obj.child.foo)
# CHECK: child.foo.loc loc("-":{{.*}}:{{.*}})
print("child.foo.loc", obj.child.get_field_loc("foo"))
# CHECK: ('Root', 'x')
print(obj.reference)

for (name, field) in obj:
  # location from om.class.field @child, %0 : !om.class.type<@Child>
  # CHECK: name: child, field: <circt.dialects.om.Object object
  # CHECK-SAME: loc: loc("-":{{.*}}:{{.*}})
  # location from om.class.field @field, %param : !om.integer
  # CHECK: name: field, field: 42
  # CHECK-SAME: loc: loc("-":{{.*}}:{{.*}})
  # location from om.class.field @reference, %sym : !om.ref
  # CHECK: name: reference, field: ('Root', 'x')
  # CHECK-SAME: loc: loc("-":{{.*}}:{{.*}})
  loc = obj.get_field_loc(name)
  print(f"name: {name}, field: {field}, loc: {loc}")

print("Check list")
# CHECK: Check list
# CHECK: ['X', 'Y']
print(list(obj.list))
for child in obj.nest.list_child:
  # CHECK: 14
  # CHECK-NEXT: 15
  print(child.foo)

obj = evaluator.instantiate("Client")
object_dict: Dict[om.Object, str] = {}
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

paths_ops = paths_class.regions[0].blocks[0].operations
# NOTE: would be nice if this supported [-1] indexing syntax
class_fields_op = paths_ops[len(paths_ops) - 1]
assert len(class_fields_op.operands)
for arg in class_fields_op.operands:
  assert isinstance(arg.type, om.PathType)

delayed = evaluator.instantiate("IntegerBinaryArithmeticObjectsDelayed")

# CHECK: 3
print(delayed.result)

# Test string and list arguments
obj = evaluator.instantiate("AppendList", "a", ["b", "c"])
# CHECK: ['a', 'b', 'c']
print(list(obj.result))

# Test string and list arguments
try:
  obj = evaluator.instantiate("AppendList", "a", [1, "b"])
except TypeError as e:
  # CHECK: List elements must be of the same type
  print(e)

try:
  obj = evaluator.instantiate("AppendList", "a", [])
except TypeError as e:
  # CHECK: Empty list is prohibited now
  print(e)

with Context() as ctx:
  circt.register_dialects(ctx)

  # Signed
  int_attr3 = om.OMIntegerAttr.get(
      IntegerAttr.get(IntegerType.get_signed(64), 42))
  # CHECK: 42
  print(str(int_attr3))

  int_attr4 = om.OMIntegerAttr.get(
      IntegerAttr.get(IntegerType.get_signed(64), -42))
  # CHECK: -42
  print(str(int_attr4))

  # Test AnyType
  any_type = Type.parse("!om.any")
  assert isinstance(any_type, om.AnyType)

  # Test ListType
  list_type = Type.parse("!om.list<!om.any>")
  assert isinstance(list_type, om.ListType)
  assert isinstance(list_type.element_type, om.AnyType)
