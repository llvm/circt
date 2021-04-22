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
    instance_builder_tests = rtl.RTLModuleOp(name="instance_builder_tests")
    one_input = rtl.RTLModuleOp(
        name="one_input",
        input_ports=[("a", i32)],
        body_builder=lambda m: rtl.OutputOp([]),
    )
    one_output = rtl.RTLModuleOp(
        name="one_output",
        output_ports=[("a", i32)],
        body_builder=lambda m: rtl.OutputOp(
            [rtl.ConstantOp(i32, IntegerAttr.get(i32, 46)).result]),
    )
    input_output = rtl.RTLModuleOp(
        name="input_output",
        input_ports=[("a", i32)],
        output_ports=[("b", i32)],
        body_builder=lambda m: rtl.OutputOp([m.entry_block.arguments[0]]),
    )

    with InsertionPoint(instance_builder_tests.add_entry_block()):
      # CHECK: unknown input port name b
      try:
        inst2 = one_input.create("inst2")
        inst2.b = None
      except AttributeError as e:
        print(e)

      # CHECK: unknown output port name b
      try:
        inst3 = one_output.create("inst3")
        inst3.b
      except AttributeError as e:
        print(e)
