# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt import esi
from circt.dialects import rtl

from mlir.ir import *
from mlir.dialects import builtin

from os import path
thisDir = path.dirname(__file__)

with Context() as ctxt, Location.unknown():
    circt.register_dialects(ctxt)
    sys = esi.System(ctxt)
    sys.load_mlir(path.join(thisDir, "esi_load1.mlir"))
    sys.load_mlir(path.join(thisDir, "esi_load2.mlir"))
    m = sys.get()

    i1 = IntegerType.get_signless(1)
    i32 = IntegerType.get_signless(32)

    with InsertionPoint(m.regions[0].blocks[0]):
        op = rtl.RTLModuleOp(
            name='MyWidget',
            input_ports=[('foo', i32), ('foo_valid', i1)],
            output_ports=[('foo_ready', i1)],
            body_builder=lambda module: rtl.OutputOp(
                [module.entry_block.arguments[1]])
        )

    esi.buildWrapper(op.operation, ["foo"])
    m.print()
    # CHECK-LABEL:  rtl.module @MyWidget_esi(%foo: !esi.channel<i32>) {
    # CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %foo, %pearl.foo_ready : i32
    # CHECK-NEXT:     %pearl.foo_ready = rtl.instance "pearl" @MyWidget(%rawOutput, %valid) : (i32, i1) -> i1
    # CHECK-NEXT:     rtl.output
    # CHECK-LABEL:  rtl.module @MyWidget(%foo: i32, %foo_valid: i1) -> (%foo_ready: i1) {
    # CHECK-NEXT:     rtl.output %foo_valid : i1

    prod = sys.lookup("IntProducer")
    assert (prod is not None)
    prod.print()
    print()  # Newline.
    # CHECK: rtl.module.extern @IntProducer(%clk: i1) -> (%ints: !esi.channel<i32>)

    acc = sys.lookup("IntAccumulator")
    assert (acc is not None)
    acc.print()
    print()  # Newline.
    # CHECK: rtl.module.extern @IntAccumulator(%clk: i1, %ints: i32, %ints_valid: i1) -> (%ints_ready: i1, %sum: i32)
