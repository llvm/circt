// RUN: circt-opt --pass-pipeline='builtin.module(any(pipeline-schedule-linear))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @pipeline(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32, i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
// CHECK:             %[[VAL_8:.*]] = hw.constant true
// CHECK:             %[[VAL_9:.*]] = comb.add %[[VAL_6]], %[[VAL_7]] {ssp.operator_type = @add1} : i32
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_7]], %[[VAL_6]] {ssp.operator_type = @add1} : i32
// CHECK:             pipeline.stage ^bb1 enable %[[VAL_8]]
// CHECK:           ^bb1:
// CHECK:             pipeline.stage ^bb2 enable %[[VAL_8]]
// CHECK:           ^bb2:
// CHECK:             %[[VAL_11:.*]] = comb.mul %[[VAL_6]], %[[VAL_9]] {ssp.operator_type = @mul2} : i32
// CHECK:             pipeline.stage ^bb3 enable %[[VAL_8]]
// CHECK:           ^bb3:
// CHECK:             pipeline.stage ^bb4 enable %[[VAL_8]]
// CHECK:           ^bb4:
// CHECK:             pipeline.stage ^bb5 enable %[[VAL_8]]
// CHECK:           ^bb5:
// CHECK:             %[[VAL_12:.*]] = comb.add %[[VAL_11]], %[[VAL_10]] {ssp.operator_type = @add1} : i32
// CHECK:             pipeline.stage ^bb6 enable %[[VAL_8]]
// CHECK:           ^bb6:
// CHECK:             pipeline.stage ^bb7 enable %[[VAL_8]]
// CHECK:           ^bb7:
// CHECK:             pipeline.return %[[VAL_12]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_13:.*]] : i32
// CHECK:         }

module {
  ssp.library @lib {
    operator_type @add1 [latency<2>]
    operator_type @mul2 [latency<3>]
  }

  hw.module @pipeline(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
    %0 = pipeline.unscheduled(%arg0, %arg1) clock %clk reset %rst
      {operator_lib = @lib}
    : (i32, i32) -> (i32) {
    ^bb0(%a0 : i32, %a1: i32):
      %0 = comb.add %a0, %a1 {ssp.operator_type = @add1} : i32
      %1 = comb.mul %a0, %0 {ssp.operator_type = @mul2} : i32
      %2 = comb.add %a1, %a0 {ssp.operator_type = @add1} : i32
      %3 = comb.add %1, %2 {ssp.operator_type = @add1} : i32
      pipeline.return %3 : i32
    }
    hw.output %0 : i32
  }
}
