// RUN: circt-opt --pass-pipeline='hw.module(pipeline.pipeline(pipeline-schedule-linear))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @pipeline(
// CHECK-SAME:       %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_4:.*]] = pipeline.pipeline(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_2]] reset %[[VAL_3]] {scheduling.operator_info = [{latency = 1 : i64, name = "comb.add"}, {latency = 2 : i64, name = "comb.mul"}]} : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             %[[VAL_8:.*]] = hw.constant true
// CHECK:             %[[VAL_9:.*]] = comb.add %[[VAL_5]], %[[VAL_6]]
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_6]], %[[VAL_5]]
// CHECK:             %[[VAL_11:.*]] = pipeline.stage when %[[VAL_7]]
// CHECK:             %[[VAL_12:.*]] = comb.mul %[[VAL_5]], %[[VAL_9]]
// CHECK:             %[[VAL_13:.*]] = pipeline.stage when %[[VAL_11]]
// CHECK:             %[[VAL_14:.*]] = pipeline.stage when %[[VAL_13]]
// CHECK:             %[[VAL_15:.*]] = comb.add %[[VAL_12]], %[[VAL_10]]
// CHECK:             %[[VAL_16:.*]] = pipeline.stage when %[[VAL_14]]
// CHECK:             pipeline.return %[[VAL_15]] valid %[[VAL_8]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_17:.*]] : i32
// CHECK:         }

hw.module @pipeline(%arg0 : i32, %arg1 : i32, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1) clock %clk reset %rst
  {
    scheduling.operator_info = [
        { name = "comb.add", latency = 1},
        { name = "comb.mul", latency = 2}
    ]
  }
   : (i32, i32) -> (i32) {
   ^bb0(%a0 : i32, %a1: i32):
    %0 = comb.add %a0, %a1 : i32
    %1 = comb.mul %a0, %0 : i32
    %2 = comb.add %a1, %a0 : i32
    %3 = comb.add %1, %2 : i32
    %c1_i1 = hw.constant 1 : i1
    pipeline.return %3 valid %c1_i1 : i32
  }
  hw.output %0 : i32
}
