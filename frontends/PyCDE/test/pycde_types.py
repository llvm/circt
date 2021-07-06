# RUN: %PYTHON% %s 2>&1 | FileCheck %s

from pycde import dim, types

# CHECK: !hw.struct<foo: i1, bar: i13>
st1 = types.struct({"foo": types.i1, "bar": types.i13})
st1.dump()
print()

# CHECK: i6
array1 = dim(types.i6)
array1.dump()
print()

# CHECK: !hw.array<12xarray<10xi6>>
array2 = dim(6, 10, 12)
array2.dump()
print()

# CHECK: !hw.typealias<@pycde::@myname1, i8>
int_alias = types.int(8, "myname1")
int_alias.dump()
print()

# CHECK: !hw.typealias<@pycde::@myname1, i8>
int_alias = types.int(8, "myname1")
int_alias.dump()
print()

# CHECK: !hw.typealias<@pycde::@myname2, i8>
int_alias = types.int(8, "myname2")
int_alias.dump()
print()

# CHECK: !hw.typealias<@pycde::@myname3, !hw.array<8xi1>>
array_alias = types.array(types.i1, 8, "myname3")
array_alias.dump()
print()

# CHECK: !hw.typealias<@pycde::@myname4, !hw.struct<a: i1, b: i1>>
struct_alias = types.struct({"a": types.i1, "b": types.i1}, "myname4")
struct_alias.dump()
print()

# CHECK: !hw.typealias<@pycde::@myname5, !hw.array<8xi1>
dim_alias = dim(1, 8, name="myname5")
dim_alias.dump()
print()
