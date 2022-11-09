// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(pipeline.pipeline(pipeline-explicit-regs)))' | FileCheck %s

// CHECK:      %[[PIPELINE_RES:.*]] = pipeline.pipeline(%arg0, %go) clock %clk reset %rst : (i32, i1) -> i32 {
// CHECK-NEXT:   ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i1):
// CHECK-NEXT:     %[[ARG0_REG:.*]], %[[S0_VALID:.*]] = pipeline.stage.register when %[[ARG1]] regs %[[ARG0]] : i32
// CHECK-NEXT:     pipeline.return %[[ARG0_REG]] valid %[[S0_VALID]] : i32
// CHECK-NEXT: }

hw.module @test1(%arg0 : i32,  %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %out = pipeline.pipeline(%arg0, %go) clock %clk reset %rst : (i32, i1) -> (i32) {
    ^bb0(%a0 : i32, %g : i1):
      %s0_valid = pipeline.stage when %g
      pipeline.return %a0 valid %s0_valid : i32
  }
  hw.output %out : i32
}

// CHECK:      %0 = pipeline.pipeline(%arg0, %go) clock %clk reset %rst : (i32, i1) -> i32 {
// CHECK-NEXT:   ^bb0(%[[ARG0]]: i32, %[[ARG1:.*]]: i1):
// CHECK-NEXT:     %[[S0_REG:.*]], %[[S0_VALID:.*]] = pipeline.stage.register when %[[ARG1]] regs %[[ARG0]] : i32
// CHECK-NEXT:     %[[ADD0_OUT:.*]] = comb.add %[[S0_REG]], %[[S0_REG]] : i32
// CHECK-NEXT:     pipeline.return %[[ADD0_OUT]] valid %[[S0_VALID]] : i32
// CHECK-NEXT: }

hw.module @test2(%arg0 : i32,  %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %out = pipeline.pipeline(%arg0, %go) clock %clk reset %rst : (i32, i1) -> (i32) {
    ^bb0(%a0 : i32, %g : i1):
      %s0_valid = pipeline.stage when %g
      %add = comb.add %a0, %a0 : i32
      pipeline.return %add valid %s0_valid : i32
  }
  hw.output %out : i32
}

// CHECK:      %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
// CHECK-NEXT:   ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i1):
// CHECK-NEXT:     %[[ADD0_OUT:.*]] = comb.add %[[ARG0]], %[[ARG1]] : i32
// CHECK-NEXT:     %[[S0_REGS:.*]]:2, %[[S0_VALID:.*]] = pipeline.stage.register when %[[ARG2]] regs %[[ADD0_OUT]], %[[ARG0]] : i32, i32
// CHECK-NEXT:     %[[ADD1_OUT:.*]] = comb.add %[[S0_REGS]]#0, %[[S0_REGS]]#1 : i32
// CHECK-NEXT:     %[[S1_REGS:.*]]:2, %[[S1_VALID:.*]] = pipeline.stage.register when %[[S0_VALID]] regs %[[ADD1_OUT]], %[[S0_REGS]]#0 : i32, i32
// CHECK-NEXT:     %[[ADD2_OUT:.*]] = comb.add %[[S1_REGS]]#0, %[[S1_REGS]]#1 : i32
// CHECK-NEXT:     pipeline.return %[[ADD2_OUT]] valid %[[S1_VALID]] : i32
// CHECK-NEXT: }

hw.module @test3(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %out = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
    ^bb0(%a0 : i32, %a1: i32, %g : i1):
      %add0 = comb.add %a0, %a1 : i32

      %s0_valid = pipeline.stage when %g
      %add1 = comb.add %add0, %a0 : i32 // %a0 is a block argument fed through a stage.

      %s1_valid = pipeline.stage when %s0_valid
      %add2 = comb.add %add1, %add0 : i32 // %add0 crosses multiple stages.

      pipeline.return %add2 valid %s1_valid : i32
  }
  hw.output %out : i32
}
