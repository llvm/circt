# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import rtl

from mlir.ir import *
from mlir.passmanager import PassManager

import sys

with Context() as ctx, Location.unknown():
    circt.register_dialects(ctx)

    i32 = IntegerType.get_signless(32)

    m = Module.create()
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

        # CHECK: rtl.module @swap(%a: i32, %b: i32) -> (%{{.+}}: i32, %{{.+}}: i32)
        # CHECK:   rtl.output %b, %a : i32, i32
        @rtl.RTLModuleOp.from_py_func(i32, i32)
        def swap(a, b):
            return b, a

        # CHECK: rtl.module @top(%a: i32, %b: i32) -> (%{{.+}}: i32, %{{.+}}: i32)
        # CHECK:   %[[a0:.+]], %[[b0:.+]] = rtl.instance "" @swap(%a, %b)
        # CHECK:   %[[a1:.+]], %[[b1:.+]] = rtl.instance "" @swap(%[[a0]], %[[b0]])
        # CHECK:   rtl.output %[[a1:.+]], %[[b1:.+]] : i32, i32
        @rtl.RTLModuleOp.from_py_func(i32, i32)
        def top(a, b):
            a, b = swap(a, b)
            a, b = swap(a, b)
            return a, b

    m.operation.print()

    # CHECK-LABEL: === Verilog ===
    print("=== Verilog ===")

    pm = PassManager.parse("rtl-legalize-names,rtl.module(rtl-cleanup)")
    pm.run(m)
    # CHECK: module MyWidget
    # CHECK: module swap
    # CHECK: module top
    circt.export_verilog(m, sys.stdout)
