# RUN: %PYTHON% %s | FileCheck %s

from pycde import dim, types, Input, Output, generator, System, Module
from pycde.types import Bits, StructType, TypeAlias, UInt
from pycde.testing import unittestmodule
from pycde.value import Struct

# CHECK: [('foo', bits1), ('bar', bits13)]
st1 = StructType({"foo": types.i1, "bar": types.i13})
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
m = System([]).mod
TypeAlias.declare_aliases(m)
TypeAlias.declare_aliases(m)
print(m)


class ExStruct(Struct):
  a: Bits(4)
  b: UInt(32)

  def get_b_plus1(self):
    return self.b + UInt(1)(1)


print(ExStruct)


# CHECK-LABEL:  msft.module @TestStruct {} (%inp1: !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>) -> (out1: ui33)
# CHECK-NEXT:     %b = hw.struct_extract %inp1["b"] {sv.namehint = "inp1__b"} : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     [[r0:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[r1:%.+]] = hwarith.add %b, [[r0]] : (ui32, ui1) -> ui33
# CHECK-NEXT:     msft.output [[r1]] : ui33
@unittestmodule()
class TestStruct(Module):
  inp1 = Input(ExStruct)
  out1 = Output(UInt(33))

  @generator
  def build(self):
    self.out1 = self.inp1.get_b_plus1()
