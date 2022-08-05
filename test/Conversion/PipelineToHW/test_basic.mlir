// RUN: circt-opt --lower-pipeline-to-hw %s | FileCheck %s

// CHECK-LABEL:  hw.module @test0(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
// CHECK-NEXT:    %0 = comb.add %arg0, %arg1 : i32
// CHECK-NEXT:    %s0_valid = seq.compreg %go, %clk : i1
// CHECK-NEXT:    %s0_reg0 = seq.compreg %0, %clk : i32
// CHECK-NEXT:    %s0_reg1 = seq.compreg %arg0, %clk : i32
// CHECK-NEXT:    %1 = comb.add %s0_reg0, %s0_reg1 : i32
// CHECK-NEXT:    %s1_valid = seq.compreg %s0_valid, %clk : i1
// CHECK-NEXT:    %s1_reg0 = seq.compreg %1, %clk : i32
// CHECK-NEXT:    %s1_reg1 = seq.compreg %s0_reg0, %clk : i32
// CHECK-NEXT:    %2 = comb.add %s1_reg0, %s1_reg1 : i32
// CHECK-NEXT:    hw.output %2 : i32
// CHECK-NEXT:  }

hw.module @test0(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %1 = comb.add %arg0_0, %arg1_1 : i32
    %regOuts:2, %valid = pipeline.stage.register when %arg2 regs %1, %arg0_0 : i32, i32
    %2 = comb.add %regOuts#0, %regOuts#1 : i32
    %regOuts_2:2, %valid_3 = pipeline.stage.register when %valid regs %2, %regOuts#0 : i32, i32
    %3 = comb.add %regOuts_2#0, %regOuts_2#1 : i32
    pipeline.return %3 valid %valid_3 : i32
  }
  hw.output %0 : i32
}
