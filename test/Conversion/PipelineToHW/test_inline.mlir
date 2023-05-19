// RUN: circt-opt --lower-pipeline-to-hw %s | FileCheck %s

// CHECK:        hw.module @test0(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
// CHECK-NEXT:     %0 = comb.add %arg0, %arg1 : i32
// CHECK-NEXT:     %p0_s0_reg0 = seq.compreg %0, %clk : i32
// CHECK-NEXT:     %p0_s0_reg1 = seq.compreg %arg0, %clk : i32
// CHECK-NEXT:     %p0_s0_valid = seq.compreg %go, %clk : i1
// CHECK-NEXT:     %1 = comb.add %p0_s0_reg0, %p0_s0_reg1 : i32
// CHECK-NEXT:     %p0_s1_reg0 = seq.compreg %1, %clk : i32
// CHECK-NEXT:     %p0_s1_reg1 = seq.compreg %p0_s0_reg0, %clk : i32
// CHECK-NEXT:     %p0_s1_valid = seq.compreg %p0_s0_valid, %clk : i1
// CHECK-NEXT:     %2 = comb.add %p0_s1_reg0, %p0_s1_reg1 : i32
// CHECK-NEXT:     hw.output %2 : i32
// CHECK-NEXT:   }
hw.module @test0(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %outputs:2, %valid = pipeline.stage ins %arg0_0, %arg1_1 enable %arg2 : (i32, i32) -> (i32, i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2, %arg3 valid %arg5 : (i32, i32)
    }
    %outputs_2:2, %valid_3 = pipeline.stage ins %outputs#0, %outputs#1 enable %valid : (i32, i32) -> (i32, i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2, %arg3 valid %arg5 : (i32, i32)
    }
    %1 = comb.add %outputs_2#0, %outputs_2#1 : i32
    pipeline.return %1 valid %valid_3 : i32
  }
  hw.output %0 : i32
}

// CHECK:       hw.module @testMultiple(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
// CHECK-NEXT:    %0 = comb.add %arg0, %arg1 : i32
// CHECK-NEXT:    %p0_s0_reg0 = seq.compreg %0, %clk : i32
// CHECK-NEXT:    %p0_s0_valid = seq.compreg %go, %clk : i1
// CHECK-NEXT:    %1 = comb.add %p0_s0_reg0, %p0_s0_reg0 : i32
// CHECK-NEXT:    %p1_s0_reg0 = seq.compreg %1, %clk : i32
// CHECK-NEXT:    %p1_s0_valid = seq.compreg %go, %clk : i1
// CHECK-NEXT:    hw.output %p1_s0_reg0 : i32
// CHECK-NEXT:  }
hw.module @testMultiple(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %output, %valid = pipeline.stage ins %arg0_0, %arg1_1 enable %arg2 : (i32, i32) -> (i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2 valid %arg5 : (i32)
    }
    pipeline.return %output valid %valid : i32
  }
  %1 = pipeline.pipeline(%0, %0, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %output, %valid = pipeline.stage ins %arg0_0, %arg1_1 enable %arg2 : (i32, i32) -> (i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2 valid %arg5 : (i32)
    }
    pipeline.return %output valid %valid : i32
  }
  hw.output %1 : i32    
}
