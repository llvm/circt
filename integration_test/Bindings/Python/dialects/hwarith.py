# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, hwarith

from circt.ir import Context, Location, InsertionPoint, IntegerType, Module

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  ui4 = IntegerType.get_unsigned(4)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      # CHECK: hwarith.constant 3 : ui4{{$}}
      hwarith.ConstantOp.create(ui4, 3)

      # CHECK: hwarith.constant 3 : ui4 : !hw.typealias<@intScope::@uintAlias, ui4>
      uint_type_alias = hw.TypeAliasType.get("intScope", "uintAlias", ui4)
      hwarith.ConstantOp.create(uint_type_alias, 3)

    hw.HWModuleOp(name="test", body_builder=build)

  print(m)
