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

  # CHECK: !rtl.array<5xi32>
  array_i32 = rtl.ArrayType.get(i32, 5)
  print(array_i32)

  # CHECK: !rtl.struct<foo: i32, bar: !rtl.array<5xi32>>
  struct = rtl.StructType.get([("foo", i32), ("bar", array_i32)])
  print(struct)

  # CHECK: !rtl.struct<baz: i32, qux: !rtl.array<5xi32>>
  struct = rtl.StructType.get([("baz", i32), ("qux", array_i32)])
  print(struct)

  m = Module.create()
  with InsertionPoint(m.body):
    # CHECK: rtl.module @MyWidget(%my_input: i32) -> (%my_output: i32)
    # CHECK:   rtl.output %my_input : i32
    op = rtl.HWModuleOp(name='MyWidget',
                         input_ports=[('my_input', i32)],
                         output_ports=[('my_output', i32)],
                         body_builder=lambda module: rtl.OutputOp(
                             [module.entry_block.arguments[0]]))

    # CHECK: rtl.module.extern @FancyThing(%input0: i32) -> (%output0: i32)
    extern = rtl.HWModuleExternOp(name="FancyThing",
                                   input_ports=[("input0", i32)],
                                   output_ports=[("output0", i32)])

    # CHECK: rtl.module @swap(%a: i32, %b: i32) -> (%{{.+}}: i32, %{{.+}}: i32)
    # CHECK:   rtl.output %b, %a : i32, i32
    @rtl.HWModuleOp.from_py_func(i32, i32)
    def swap(a, b):
      return b, a

    # CHECK: rtl.module @top(%a: i32, %b: i32) -> (%{{.+}}: i32, %{{.+}}: i32)
    # CHECK:   %[[a0:.+]], %[[b0:.+]] = rtl.instance "" @swap(%a, %b)
    # CHECK:   %[[a1:.+]], %[[b1:.+]] = rtl.instance "" @swap(%[[a0]], %[[b0]])
    # CHECK:   rtl.output %[[a1:.+]], %[[b1:.+]] : i32, i32
    @rtl.HWModuleOp.from_py_func(i32, i32)
    def top(a, b):
      a, b = swap(a, b)
      a, b = swap(a, b)
      return a, b

    one_input = rtl.HWModuleOp(
        name="one_input",
        input_ports=[("a", i32)],
        body_builder=lambda m: rtl.OutputOp([]),
    )
    two_inputs = rtl.HWModuleOp(
        name="two_inputs",
        input_ports=[("a", i32), ("b", i32)],
        body_builder=lambda m: rtl.OutputOp([]),
    )
    one_output = rtl.HWModuleOp(
        name="one_output",
        output_ports=[("a", i32)],
        body_builder=lambda m: rtl.OutputOp(
            [rtl.ConstantOp(i32, IntegerAttr.get(i32, 46)).result]),
    )

    # CHECK-LABEL: rtl.module @instance_builder_tests
    def instance_builder_body(module):
      # CHECK: %[[INST1_RESULT:.+]] = rtl.instance "inst1" @one_output()
      inst1 = one_output.create(module, "inst1")

      # CHECK: rtl.instance "inst2" @one_input(%[[INST1_RESULT]])
      inst2 = one_input.create(module, "inst2", {"a": inst1.a})

      # CHECK: rtl.instance "inst4" @two_inputs(%[[INST1_RESULT]], %[[INST1_RESULT]])
      inst4 = two_inputs.create(module, "inst4", {"a": inst1.a})
      inst4.set_input_port("b", inst1.a)

      # CHECK: %[[INST5_RESULT:.+]] = rtl.instance "inst5" @MyWidget(%[[INST5_RESULT]])
      inst5 = op.create(module, "inst5")
      inst5.set_input_port("my_input", inst5.my_output)

      # CHECK: rtl.instance "inst6" {{.*}} {BANKS = 2 : i64}
      one_input.create(module, "inst6", {"a": inst1.a}, parameters={"BANKS": 2})

      rtl.OutputOp([])

    instance_builder_tests = rtl.HWModuleOp(name="instance_builder_tests",
                                             body_builder=instance_builder_body)

    # CHECK: rtl.module @block_args_test(%[[PORT_NAME:.+]]: i32) ->
    # CHECK: rtl.output %[[PORT_NAME]]
    rtl.HWModuleOp(name="block_args_test",
                    input_ports=[("foo", i32)],
                    output_ports=[("bar", i32)],
                    body_builder=lambda module: rtl.OutputOp([module.foo]))

  print(m)

  # CHECK-LABEL: === Verilog ===
  print("=== Verilog ===")

  pm = PassManager.parse("rtl-legalize-names,rtl.module(rtl-cleanup)")
  pm.run(m)
  # CHECK: module MyWidget
  # CHECK: external module FancyThing
  # CHECK: module swap
  # CHECK: module top
  circt.export_verilog(m, sys.stdout)
