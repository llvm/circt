# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, pipeline, seq
from circt.ir import Context, Location, InsertionPoint, IntegerType, Module

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i32 = IntegerType.get_signless(32)
  i1 = IntegerType.get_signless(1)
  clk = seq.ClockType.get(ctx)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      # CHECK: hw.constant
      const = hw.ConstantOp.create(i32, 5)

      # CHECK: pipeline.latency
      latency_op = pipeline.LatencyOp([i32], 2)
      block = latency_op.body.blocks.append()
      with InsertionPoint(block):
        # CHECK: pipeline.latency.return
        pipeline.LatencyReturnOp([const])

      # CHECK: hw.output
      hw.OutputOp([latency_op.results[0]])

    hw.HWModuleOp(name="test",
                  input_ports=[("clk", clk), ("go", i1)],
                  body_builder=build)

  print(m)
