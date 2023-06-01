// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(pipeline.pipeline(pipeline-explicit-regs)))' | FileCheck %s

// CHECK:      %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
// CHECK-NEXT:   ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i1):
// CHECK-NEXT:     %[[ADD0_OUT:.*]] = comb.add %[[ARG0]], %[[ARG1]] : i32
// CHECK-NEXT:     %[[S0_REGS:.*]]:2, %[[S0_VALID:.*]] = pipeline.stagesep.reg enable %[[ARG2]] regs %[[ADD0_OUT]], %[[ARG0]] : i32, i32
// CHECK-NEXT:     %[[ADD1_OUT:.*]] = comb.add %[[S0_REGS]]#0, %[[S0_REGS]]#1 : i32
// CHECK-NEXT:     %[[S1_REGS:.*]]:2, %[[S1_VALID:.*]] = pipeline.stagesep.reg enable %[[S0_VALID]] regs %[[ADD1_OUT]], %[[S0_REGS]]#0 : i32, i32
// CHECK-NEXT:     %[[ADD2_OUT:.*]] = comb.add %[[S1_REGS]]#0, %[[S1_REGS]]#1 : i32
// CHECK-NEXT:     pipeline.return %[[ADD2_OUT]] valid %[[S1_VALID]] : i32
// CHECK-NEXT: }

hw.module @test3(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %out = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
    ^bb0(%a0 : i32, %a1: i32, %g : i1):
      %add0 = comb.add %a0, %a1 : i32

      %s0_valid = pipeline.stagesep enable %g
      %add1 = comb.add %add0, %a0 : i32 // %a0 is a block argument fed through a stage.

      %s1_valid = pipeline.stagesep enable %s0_valid
      %add2 = comb.add %add1, %add0 : i32 // %add0 crosses multiple stages.

      pipeline.return %add2 valid %s1_valid : i32
  }
  hw.output %out : i32
}

