# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import rtl

from mlir.ir import *
from mlir.dialects import builtin

with Context() as ctx, Location.unknown():
    circt.register_dialects(ctx)

    i32 = IntegerType.get_signless(32)

    m = builtin.ModuleOp()
    with InsertionPoint(m.body):
        # CHECK: rtl.module @MyWidget(%my_input: i32) -> (%my_output: i32)
        # CHECK:   rtl.output %my_input : i32
        op = rtl.RTLModuleOp(
            name='MyWidget',
            input_ports=[('my_input', i32)],
            output_ports=[('my_output', i32)],
            body_builder=lambda module: rtl.OutputOp(
                [module.entry_block.arguments[0]])
        )

    m.print()
