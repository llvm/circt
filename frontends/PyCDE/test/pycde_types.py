# RUN: %PYTHON% %s 2>&1 | FileCheck %s

from pycde import types

st1 = types.struct({"foo": types.i1, "bar": types.i13})
print(st1.get_fields())
# CHECK: [('foo', Type(i1)), ('bar', Type(i13))]
