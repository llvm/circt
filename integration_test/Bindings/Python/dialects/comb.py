# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.design_entry import connect
from circt.dialects import comb, hw

from mlir.ir import Context, Location, InsertionPoint, IntegerType, IntegerAttr, Module

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i32 = IntegerType.get_signless(32)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      # CHECK: %[[CONST:.+]] = hw.constant 1 : i32
      const = hw.ConstantOp(i32, IntegerAttr.get(i32, 1))

      # CHECK: comb.divs %[[CONST]], %[[CONST]]
      comb.DivSOp.create(i32, {"lhs": const.result, "rhs": const.result})
      # CHECK: comb.divs %[[CONST]], %[[CONST]]
      divs = comb.DivSOp.create(i32)
      connect(divs.lhs, const.result)
      connect(divs.rhs, const.result)

    hw.HWModuleOp(name="test", body_builder=build)

  print(m)
