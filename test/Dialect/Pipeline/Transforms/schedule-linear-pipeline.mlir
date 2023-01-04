// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(pipeline.pipeline(pipeline-schedule-linear)))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @pipeline(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_4:.*]] = pipeline.pipeline(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_2]] reset %[[VAL_3]] {operator_lib = @lib} : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             %[[VAL_9:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] {ssp.operator_type = @add1} : i32
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_6]], %[[VAL_5]] {ssp.operator_type = @add1} : i32
// CHECK:             %[[VAL_11:.*]] = pipeline.stage when %[[VAL_7]]
// CHECK:             %[[VAL_12:.*]] = pipeline.stage when %[[VAL_11]]
// CHECK:             %[[VAL_13:.*]] = comb.mul %[[VAL_5]], %[[VAL_9]] {ssp.operator_type = @mul2} : i32
// CHECK:             %[[VAL_14:.*]] = pipeline.stage when %[[VAL_12]]
// CHECK:             %[[VAL_15:.*]] = pipeline.stage when %[[VAL_14]]
// CHECK:             %[[VAL_16:.*]] = pipeline.stage when %[[VAL_15]]
// CHECK:             %[[VAL_17:.*]] = comb.add %[[VAL_13]], %[[VAL_10]] {ssp.operator_type = @add1} : i32
// CHECK:             %[[VAL_18:.*]] = pipeline.stage when %[[VAL_16]]
// CHECK:             %[[VAL_19:.*]] = pipeline.stage when %[[VAL_18]]
// CHECK:             pipeline.return %[[VAL_17]] valid %[[VAL_19]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_20:.*]] : i32
// CHECK:         }

module {
  ssp.library @lib {
    operator_type @add1 [latency<2>]
    operator_type @mul2 [latency<3>]
  }

  hw.module @pipeline(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
    %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst
      {operator_lib = @lib}
    : (i32, i32) -> (i32) {
    ^bb0(%a0 : i32, %a1: i32):
      %0 = comb.add %a0, %a1 {ssp.operator_type = @add1} : i32
      %1 = comb.mul %a0, %0 {ssp.operator_type = @mul2} : i32
      %2 = comb.add %a1, %a0 {ssp.operator_type = @add1} : i32
      %3 = comb.add %1, %2 {ssp.operator_type = @add1} : i32
      %c1_i1 = hw.constant 1 : i1
      pipeline.return %3 valid %c1_i1 : i32
    }
    hw.output %0 : i32
  }
}
