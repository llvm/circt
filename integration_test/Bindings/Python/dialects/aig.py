# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import aig, hw
from circt.ir import Context, Location, Module, InsertionPoint, IntegerType

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  m = Module.create()
  with InsertionPoint(m.body):
    # Test basic AIG dialect functionality
    i1 = IntegerType.get_signless(1)

    # Create a simple hardware module with AIG operations
    def build_module(module):
      a, b = module.entry_block.arguments

      result = aig.and_inv([a, b], [False, True])

      hw.OutputOp([result])

    hw.HWModuleOp(name="test_aig",
                  input_ports=[("a", i1), ("b", i1)],
                  output_ports=[("out", i1)],
                  body_builder=build_module)
    # CHECK-LABEL: AIG dialect registration and basic operations successful!
    print("AIG dialect registration and basic operations successful!")
    # Test aig.and_inv operation
    # CHECK: %{{.*}} = aig.and_inv %a, not %b : i1
    print(m)
