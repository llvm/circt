# RUN: %PYTHON% %s | FileCheck %s

from pycde import dim, types
from pycde.types import Bits, Struct, TypeAlias
from pycde.circt.ir import Module

# CHECK: [('foo', bits1), ('bar', bits13)]
st1 = Struct({"foo": types.i1, "bar": types.i13})
print(st1.fields)
# CHECK: bits1
print(st1.foo)

array1 = dim(types.ui6)
# CHECK: uint6
print(array1)

array2 = dim(6, 10, 12)
# CHECK: [12][10]bits6
print(array2)

int_alias = TypeAlias(Bits(8), "myname1")
# CHECK: myname1
print(int_alias)
assert int_alias == types.int(8, "myname1")

# CHECK: struct { a: bits1, b: sint1}
struct = types.struct({"a": types.i1, "b": types.si1})
print(struct)

dim_alias = dim(1, 8, name="myname5")

# CHECK: hw.type_scope @pycde
# CHECK: hw.typedecl @myname1 : i8
# CHECK: hw.typedecl @myname5 : !hw.array<8xi1>
# CHECK-NOT: hw.typedecl @myname1
# CHECK-NOT: hw.typedecl @myname5
m = Module.create()
TypeAlias.declare_aliases(m)
TypeAlias.declare_aliases(m)
print(m)
