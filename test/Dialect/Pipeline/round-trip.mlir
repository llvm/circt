// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @unscheduled1
hw.module @unscheduled1(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.unscheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = pipeline.latency 2 -> (i32) {
      %1 = comb.add %a0, %a1 : i32
      pipeline.latency.return %1 : i32
    }
    %c1_i1 = hw.constant 1 : i1
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL: hw.module @scheduled1
hw.module @scheduled1(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = comb.add %a0, %a1 : i32
    %c1_i1 = hw.constant 1 : i1
    pipeline.stage ^bb1 enable %c1_i1

   ^bb1:
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}


// CHECK-LABEL: hw.module @scheduled2
hw.module @scheduled2(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = comb.add %a0, %a1 : i32
    %c1_i1 = hw.constant 1 : i1
    pipeline.stage ^bb1 regs(%0, %c1_i1 : i32, i1) enable %c1_i1

   ^bb1(%s0_0 : i32, %s0_valid : i1):
    pipeline.return %s0_0 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL: hw.module @scheduledWithPassthrough
hw.module @scheduledWithPassthrough(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0, %1 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst : (i32, i32) -> (i32, i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = comb.add %a0, %a1 : i32
    %c1_i1 = hw.constant 1 : i1
    pipeline.stage ^bb1 regs(%0, %c1_i1 : i32, i1) pass(%a1 : i32) enable %c1_i1

   ^bb1(%s0_0 : i32, %s0_valid : i1, %s0_pass_a1 : i32):
    pipeline.return %s0_0, %s0_pass_a1 : i32, i32
  }
  hw.output %0 : i32
}
