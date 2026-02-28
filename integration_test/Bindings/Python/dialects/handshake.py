# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.dialects import hw, handshake
from circt.ir import Context, Location, Module, InsertionPoint, IntegerAttr, IntegerType

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)
  m = Module.create()
  with InsertionPoint(m.body):
    op = handshake.FuncOp.create("foo", [("a", IntegerType.get_signless(8))],
                                 [("x", IntegerType.get_signless(1))])
    # CHECK: handshake.func @foo(i8, ...) -> i1 attributes {argNames = ["a"], resNames = ["x"]}
    print(m)
