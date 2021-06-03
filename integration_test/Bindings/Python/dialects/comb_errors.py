# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import comb, hw

from mlir.ir import Context, Location, InsertionPoint, IntegerType, IntegerAttr, Module

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i32 = IntegerType.get_signless(32)
  i31 = IntegerType.get_signless(31)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      const1 = hw.ConstantOp(i32, IntegerAttr.get(i32, 1))
      const2 = hw.ConstantOp(i31, IntegerAttr.get(i31, 1))

      # CHECK: expected same input port types, but received [Type(i32), Type(i31)]
      try:
        comb.DivSOp.create(const1.result, const2.result)
      except TypeError as e:
        print(e)

      # CHECK: result type must be specified
      try:
        comb.DivSOp.create()
      except TypeError as e:
        print(e)

    hw.HWModuleOp(name="test", body_builder=build)
