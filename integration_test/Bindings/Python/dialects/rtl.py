# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import rtl

from mlir.ir import *
from mlir.dialects import std

with Context() as ctx, Location.unknown():
    circt.register_dialects(ctx)
    m = Module.create()
    with InsertionPoint.at_block_terminator(m.body):
        i32 = IntegerType.get_signless(32)
        a50 = IntegerAttr.get(i32, 50)
        std.ConstantOp(i32, a50)
        rtl.ConstantOp(i32, a50)

# CHECK: module
m.operation.print()
