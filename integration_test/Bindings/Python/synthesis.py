# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, comb
from circt.ir import Context, Location, Module, InsertionPoint, IntegerType
from circt.passmanager import PassManager

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    i4 = IntegerType.get_signless(4)

    # Create a simple hardware module with AIG operations
    def build_module(module):
      a, b = module.entry_block.arguments
      hw.OutputOp([comb.mul([a, b])])

    hw.HWModuleOp(
        name="foo",
        input_ports=[("a", i4), ("b", i4)],
        output_ports=[("out", i4)],
        body_builder=build_module,
    )

  # Check that the synthesis pipeline is registered.
  pm = PassManager.parse(
      "builtin.module(hw.module(synthesis-aig-lowering-pipeline, "
      "synthesis-aig-optimization-pipeline))")
  pm.run(m.operation)
  # CHECK: hw.module @foo(
  # CHECK-NOT: comb.mul
  # CHECK: aig.and_inv
  print(m.operation)
