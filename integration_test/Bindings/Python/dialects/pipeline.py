# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt

from circt.ir import Context, Module
from circt import passmanager

with Context() as ctx:
  circt.register_dialects(ctx)
  mod = Module.parse("""
    hw.module @testSingle(in %arg0: i32, in %arg1: i32,
                      in %go: i1, in %clk: !seq.clock, in %rst: i1,
                      out out0: i32, out out1: i1) {
      %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1)
                      clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable)
                      -> (out: i32){
        %1 = comb.sub %a0,%a1 : i32
        pipeline.stage ^bb1 regs(%1 : i32, %a0 : i32)
      ^bb1(%6: i32, %7: i32, %s1_enable : i1):  // pred: ^bb1
        %8 = comb.add %6, %7 : i32
        pipeline.return %8 : i32
      }
      hw.output %0#0, %0#1 : i32, i1
    }
  """)

  pm = passmanager.PassManager.parse(
      "builtin.module(pipeline-explicit-regs,lower-pipeline-to-hw)")
  pm.run(mod.operation)

  # CHECK: hw.module @testSingle
  # CHECK: %[[SUB:.+]] = comb.sub %arg0, %arg1
  # CHECK: %{{.+}} = seq.compreg sym @p0_stage0_reg0 %[[SUB]], %clk
  # CHECK: %{{.+}} = seq.compreg sym @p0_stage0_reg1 %arg0, %clk
  # CHECK: %{{.+}} = seq.compreg sym @p0_stage1_enable %go, %clk reset %rst
  # CHECK: comb.add
  # CHECK: hw.output
  print(mod)
