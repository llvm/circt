// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   hw.module @unscheduled1(
// CHECK-SAME:         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_4:.*]] = pipeline.unscheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_2]] reset %[[VAL_3]] : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_8:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:               pipeline.latency.return %[[VAL_8]] : i32
// CHECK:             }
// CHECK:             %[[VAL_9:.*]] = hw.constant true
// CHECK:             pipeline.return %[[VAL_10:.*]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_11:.*]] : i32
// CHECK:         }
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

// CHECK-LABEL:   hw.module @scheduled1(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_4:.*]] = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_2]] reset %[[VAL_3]] : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_8:.*]] = hw.constant true
// CHECK:             pipeline.stage ^bb1 enable %[[VAL_8]]
// CHECK:           ^bb1:
// CHECK:             pipeline.return %[[VAL_7]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_9:.*]] : i32
// CHECK:         }
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


// CHECK-LABEL:   hw.module @scheduled2(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_4:.*]] = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_2]] reset %[[VAL_3]] : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_8:.*]] = hw.constant true
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_7]], %[[VAL_8]] : i32, i1) enable %[[VAL_8]]
// CHECK:           ^bb1(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_11:.*]] : i32
// CHECK:         }
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

// CHECK-LABEL:   hw.module @scheduledWithPassthrough(
// CHECK-SAME:              %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_4:.*]]:2 = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_2]] reset %[[VAL_3]] : (i32, i32) -> (i32, i32) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_8:.*]] = hw.constant true
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_7]], %[[VAL_8]] : i32, i1) pass(%[[VAL_6]] : i32) enable %[[VAL_8]]
// CHECK:           ^bb1(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i1, %[[VAL_11:.*]]: i32):
// CHECK:             pipeline.return %[[VAL_9]], %[[VAL_11]] : i32, i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_12:.*]]#0 : i32
// CHECK:         }
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

// CHECK-LABEL:   hw.module @withStall(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_4:.*]] = pipeline.scheduled(%[[VAL_0]]) stall %[[VAL_1]] clock %[[VAL_2]] reset %[[VAL_3]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32):
// CHECK:             pipeline.return %[[VAL_5]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_6:.*]] : i32
// CHECK:         }
hw.module @withStall(%arg0 : i32, %stall : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.scheduled(%arg0) stall %stall clock %clk reset %rst : (i32) -> (i32) {
   ^bb0(%a0 : i32):
    pipeline.return %a0 : i32
  }
  hw.output %0 : i32
}
