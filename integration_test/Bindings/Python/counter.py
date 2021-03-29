# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import comb, rtl, sv

from mlir.ir import *
from mlir.dialects import builtin

with Context() as ctx, Location.unknown():
    circt.register_dialects(ctx)

    i1 = IntegerType.get_signless(32)
    i8 = IntegerType.get_signless(32)

    m = builtin.ModuleOp()
    with InsertionPoint(m.body):
        # CHECK-LABEL: rtl.module @counter
        @rtl.RTLModuleOp.from_py_func(i1, i1)
        def counter(clk, rstn):
            counter_wire = sv.WireOp('counter', i8)
            return

    m.print()
