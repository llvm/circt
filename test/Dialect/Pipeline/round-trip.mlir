// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   hw.module @unscheduled1(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.unscheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_3]] reset %[[VAL_4]] go %[[VAL_2]] : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i1):
// CHECK:             %[[VAL_10:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_11:.*]] = comb.add %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:               pipeline.latency.return %[[VAL_11]] : i32
// CHECK:             }
// CHECK:             pipeline.return %[[VAL_12:.*]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_13:.*]] : i32
// CHECK:         }
hw.module @unscheduled1(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.unscheduled(%arg0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32, %s0_valid : i1):
    %0 = pipeline.latency 2 -> (i32) {
      %1 = comb.add %a0, %a1 : i32
      pipeline.latency.return %1 : i32
    }
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @scheduled1(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_3]] reset %[[VAL_4]] go %[[VAL_2]] : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i1):
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             pipeline.stage ^bb1
// CHECK:           ^bb1(%[[VAL_11:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_12:.*]] : i32
// CHECK:         }
hw.module @scheduled1(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32, %s0_valid : i1):
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1

   ^bb1(%s1_valid : i1):
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}


// CHECK-LABEL:   hw.module @scheduled2(
// CHECK-SAME:               %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_3]] reset %[[VAL_4]] go %[[VAL_2]] : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i1):
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_10]] : i32)
// CHECK:           ^bb1(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_11]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_13:.*]] : i32
// CHECK:         }
hw.module @scheduled2(%arg0 : i32, %arg1 : i32, %g : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst go %g : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32, %s0_valid : i1):
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1 regs(%0 : i32)

   ^bb1(%s0_0 : i32, %s1_valid : i1):
    pipeline.return %s0_0 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @scheduledWithPassthrough(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]]:2, %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_3]] reset %[[VAL_4]] go %[[VAL_2]] : (i32, i32) -> (i32, i32) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i1):
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_10]] : i32) pass(%[[VAL_8]] : i32)
// CHECK:           ^bb1(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_11]], %[[VAL_12]] : i32, i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_14:.*]]#0 : i32
// CHECK:         }
hw.module @scheduledWithPassthrough(%arg0 : i32, %arg1 : i32, %g : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:3 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst go %g : (i32, i32) -> (i32, i32) {
   ^bb0(%a0 : i32, %a1: i32, %s0_valid : i1):
    %0 = comb.add %a0, %a1 : i32
    pipeline.stage ^bb1 regs(%0 : i32) pass(%a1 : i32)

   ^bb1(%s0_0 : i32, %s0_pass_a1 : i32, %s1_valid : i1):
    pipeline.return %s0_0, %s0_pass_a1 : i32, i32
  }
  hw.output %0#0 : i32
}

// CHECK-LABEL:   hw.module @withStall(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_0]]) stall %[[VAL_1]] clock %[[VAL_3]] reset %[[VAL_4]] go %[[VAL_2]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_7]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_9:.*]] : i32
// CHECK:         }
hw.module @withStall(%arg0 : i32, %stall : i1, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%arg0) stall %stall clock %clk reset %rst go %go : (i32) -> (i32) {
   ^bb0(%a0 : i32, %s0_valid : i1):
    pipeline.return %a0 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @withMultipleRegs(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_0]]) stall %[[VAL_1]] clock %[[VAL_3]] reset %[[VAL_4]] go %[[VAL_2]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i1):
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_7]] : i32, %[[VAL_7]] : i32)
// CHECK:           ^bb1(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_12:.*]] : i32
// CHECK:         }
hw.module @withMultipleRegs(%arg0 : i32, %stall : i1, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%arg0) stall %stall clock %clk reset %rst go %go : (i32) -> (i32) {
   ^bb0(%a0 : i32, %s0_valid : i1):
    pipeline.stage ^bb1 regs(%a0 : i32, %a0 : i32)

   ^bb1(%0 : i32, %1 : i32, %s1_valid : i1):
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @withClockGates(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_0]]) stall %[[VAL_1]] clock %[[VAL_3]] reset %[[VAL_4]] go %[[VAL_2]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i1):
// CHECK:             %[[VAL_9:.*]] = hw.constant true
// CHECK:             %[[VAL_10:.*]] = hw.constant true
// CHECK:             %[[VAL_11:.*]] = hw.constant true
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_7]] : i32 gated by {{\[}}%[[VAL_9]]], %[[VAL_7]] : i32, %[[VAL_7]] : i32 gated by {{\[}}%[[VAL_10]], %[[VAL_11]]])
// CHECK:           ^bb1(%[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_12]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_16:.*]] : i32
// CHECK:         }
hw.module @withClockGates(%arg0 : i32, %stall : i1, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0:2 = pipeline.scheduled(%arg0) stall %stall clock %clk reset %rst go %go : (i32) -> (i32) {
   ^bb0(%a0 : i32, %s0_valid : i1):
    %true1 = hw.constant true
    %true2 = hw.constant true
    %true3 = hw.constant true

    pipeline.stage ^bb1 regs(%a0 : i32 gated by [%true1], %a0 : i32, %a0 : i32 gated by [%true2, %true3])

   ^bb1(%0 : i32, %1 : i32, %2 : i32, %s1_valid : i1):
    pipeline.return %0 : i32
  }
  hw.output %0 : i32
}
