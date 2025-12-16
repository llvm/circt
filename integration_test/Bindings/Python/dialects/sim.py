# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import hw, sim
from circt.ir import (Context, Location, Module, InsertionPoint, IntegerAttr,
                      IntegerType, StringAttr)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    i1 = IntegerType.get_signless(1)
    true = hw.ConstantOp(IntegerAttr.get(i1, 1))
    fmtOp = sim.FormatLiteralOp(StringAttr.get("foo"))
    # CHECK: sim.fmt.literal "foo"
    print(fmtOp)
