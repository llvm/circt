// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @retimeable1
hw.module @retimeable1(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = comb.add %a0, %a1 : i32
    %c1_i1 = hw.constant 1 : i1
    %s0_valid = pipeline.stagesep enable %c1_i1
    pipeline.return %0 valid %s0_valid : i32
  }
  hw.output %0 : i32
}


// CHECK-LABEL: hw.module @retimeable3
hw.module @retimeable3(%arg0 : !esi.channel<i32>, %arg1 : !esi.channel<i32>, %clk : i1, %rst: i1) -> (out: !esi.channel<i32>) {
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst : (!esi.channel<i32>, !esi.channel<i32>) -> (!esi.channel<i32>) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = comb.add %a0, %a1 : i32
    %c1_i1 = hw.constant 1 : i1
    %s0_valid = pipeline.stagesep enable %c1_i1
    pipeline.return %0 valid %s0_valid : i32
  }
  hw.output %0 : !esi.channel<i32>
}

// CHECK-LABEL: hw.module @stage
hw.module @stage(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32, %gogo : i1):
   %r_0, %s0_valid = pipeline.stage ins %a0, %a1 enable %gogo : (i32, i32) -> (i32) {
      ^bb0(%in0 : i32, %in1: i32, %valid : i1):
      %c1_i1 = hw.constant 1 : i1
      %0 = comb.add %in0, %in1 : i32
      pipeline.stage.return regs %0 valid %valid : (i32)
   }
    pipeline.return %r_0 valid %s0_valid : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL: hw.module @stageWithPassthrough
hw.module @stageWithPassthrough(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0, %1 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32, i32) {
   ^bb0(%a0 : i32, %a1: i32, %gogo : i1):
   %r_0, %pass_1, %s0_valid = pipeline.stage ins %a0, %a1 enable %gogo : (i32, i32) -> (i32, i32) {
      ^bb0(%in0 : i32, %in1: i32, %valid : i1):
      %c1_i1 = hw.constant 1 : i1
      %0 = comb.add %in0, %in1 : i32
      pipeline.stage.return regs %0 passthroughs %in1 valid %valid : (i32), (i32)
   }
    pipeline.return %r_0, %pass_1 valid %s0_valid : i32, i32
  }
  hw.output %0 : i32
}
