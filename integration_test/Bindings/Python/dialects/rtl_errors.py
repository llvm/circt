# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw

from mlir.ir import *
from mlir.passmanager import PassManager

import sys

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i32 = IntegerType.get_signless(32)

  m = Module.create()
  with InsertionPoint(m.body):
    one_input = hw.HWModuleOp(
        name="one_input",
        input_ports=[("a", i32)],
        body_builder=lambda m: hw.OutputOp([]),
    )
    one_output = hw.HWModuleOp(
        name="one_output",
        output_ports=[("a", i32)],
        body_builder=lambda m: hw.OutputOp(
            [hw.ConstantOp(i32, IntegerAttr.get(i32, 46)).result]),
    )
    input_output = hw.HWModuleOp(
        name="input_output",
        input_ports=[("a", i32)],
        output_ports=[("b", i32)],
        body_builder=lambda m: hw.OutputOp([m.entry_block.arguments[0]]),
    )

    def instance_builder_body(module):
      constant_value = one_output.create(module, "inst1").a

      # CHECK: unknown input port name b
      try:
        inst2 = one_input.create(module, "inst2", {"a": constant_value})
        inst2.set_input_port("b", None)
      except AttributeError as e:
        print(e)

      # CHECK: unknown port name b
      try:
        inst3 = one_output.create(module, "inst3")
        inst3.b
      except AttributeError as e:
        print(e)

      # CHECK: unknown input port name nonexistant_port
      try:
        module.nonexistant_port
      except AttributeError as e:
        print(e)

      # Note, the error here is actually caught and printed below.
      # CHECK: Uninitialized ports remain in circuit!
      # CHECK: Port:     %[[PORT_NAME:.+]]
      # CHECK: Module:   hw.module @one_input(%[[PORT_NAME]]: i32)
      # CHECK: Instance: hw.instance "inst1" @one_input({{.+}})
      inst1 = one_input.create(module, "inst1")

    try:
      instance_builder_tests = hw.HWModuleOp(
          name="instance_builder_tests", body_builder=instance_builder_body)
    except RuntimeError as e:
      print(e)
