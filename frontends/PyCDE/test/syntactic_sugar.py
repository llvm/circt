# RUN: %PYTHON% %s | FileCheck %s

from pycde import types, dim, obj_to_value, System


class Top(System):
  BarType = types.struct({"foo": types.i12}, "bar")
  inputs = []
  outputs = []

  def build(self, top):
    obj_to_value({"foo": 7}, types.struct({"foo": types.i12}))
    obj_to_value([42, 45], dim(types.i8, 2))
    obj_to_value(5, types.i8)

    Top.BarType.create({"foo": 7})


top = Top()
top.print()
# CHECK-LABEL: hw.module @top()
# CHECK:  %c7_i12 = hw.constant 7 : i12
# CHECK:  %0 = hw.struct_create (%c7_i12) : !hw.struct<foo: i12>
# CHECK:  %c42_i8 = hw.constant 42 : i8
# CHECK:  %c45_i8 = hw.constant 45 : i8
# CHECK:  %1 = hw.array_create %c45_i8, %c42_i8 : i8
# CHECK:  %c5_i8 = hw.constant 5 : i8
# CHECK:  %c7_i12_0 = hw.constant 7 : i12
# CHECK:  %2 = hw.struct_create (%c7_i12_0) : !hw.struct<foo: i12>
# CHECK:  %3 = hw.bitcast %2 : (!hw.struct<foo: i12>) -> !hw.typealias<@pycde::@bar, !hw.struct<foo: i12>>
