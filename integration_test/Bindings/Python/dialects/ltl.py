# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import hw, ltl
from circt.ir import Context, Location, Module, InsertionPoint, IntegerAttr, IntegerType

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    i1 = IntegerType.get_signless(1)
    true = hw.ConstantOp(IntegerAttr.get(i1, 1))
    andOp = ltl.AndOp([true, true])
    # CHECK: ltl.and %true, %true : i1, i1
    print(andOp)
