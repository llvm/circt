# RUN: %PYTHON% %s 2>&1 | FileCheck %s

from pycde import dim, types

# CHECK: i6
array1 = dim(types.i6)
array1.dump()
print()

# CHECK: !hw.array<12xarray<10xi6>>
array2 = dim(6, 10, 12)
array2.dump()
print()
