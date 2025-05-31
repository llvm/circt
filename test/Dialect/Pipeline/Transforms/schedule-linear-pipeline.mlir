// RUN: circt-opt --pass-pipeline='builtin.module(any(pipeline-schedule-linear))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @pipeline(in 
// CHECK-SAME:      %[[ARG0:.*]] : i32, in %[[ARG1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLK:.*]] : !seq.clock, in %[[RST:.*]] : i1, out out : i32) {
// CHECK:           %[[VAL_0:.*]], %[[VAL_1:.*]] = pipeline.scheduled(%[[VAL_2:.*]] : i32 = %[[ARG0]], %[[VAL_3:.*]] : i32 = %[[ARG1]]) clock(%[[CLK]]) reset(%[[RST]]) go(%[[GO]]) entryEn(%[[VAL_4:.*]])  -> (out : i32) {
// CHECK:             %[[VAL_5:.*]] = comb.add %[[VAL_2]], %[[VAL_3]] {ssp.operator_type = @add1} : i32
// CHECK:             %[[VAL_6:.*]] = comb.add %[[VAL_3]], %[[VAL_2]] {ssp.operator_type = @add1} : i32
// CHECK:             pipeline.stage ^bb1
// CHECK:           ^bb1(%[[VAL_7:.*]]: i1):
// CHECK:             pipeline.stage ^bb2
// CHECK:           ^bb2(%[[VAL_8:.*]]: i1):
// CHECK:             %[[VAL_9:.*]] = pipeline.src %[[VAL_2]] : i32
// CHECK:             %[[VAL_10:.*]] = pipeline.src %[[VAL_5]] : i32
// CHECK:             %[[VAL_11:.*]] = comb.mul %[[VAL_9]], %[[VAL_10]] {ssp.operator_type = @mul2} : i32
// CHECK:             pipeline.stage ^bb3
// CHECK:           ^bb3(%[[VAL_12:.*]]: i1):
// CHECK:             pipeline.stage ^bb4
// CHECK:           ^bb4(%[[VAL_13:.*]]: i1):
// CHECK:             pipeline.stage ^bb5
// CHECK:           ^bb5(%[[VAL_14:.*]]: i1):
// CHECK:             %[[VAL_15:.*]] = pipeline.src %[[VAL_11]] : i32
// CHECK:             %[[VAL_16:.*]] = pipeline.src %[[VAL_6]] : i32
// CHECK:             %[[VAL_17:.*]] = comb.add %[[VAL_15]], %[[VAL_16]] {ssp.operator_type = @add1} : i32
// CHECK:             pipeline.stage ^bb6
// CHECK:           ^bb6(%[[VAL_18:.*]]: i1):
// CHECK:             pipeline.stage ^bb7
// CHECK:           ^bb7(%[[VAL_19:.*]]: i1):
// CHECK:             %[[VAL_20:.*]] = pipeline.src %[[VAL_17]] : i32
// CHECK:             pipeline.return %[[VAL_20]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_21:.*]] : i32
// CHECK:         }

module {
  ssp.library @lib {
    operator_type @add1 [latency<2>]
    operator_type @mul2 [latency<3>]
  }

  hw.module @pipeline(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
    %0:2 = pipeline.unscheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable)
      {operator_lib = @lib} -> (out: i32) {
      %0 = comb.add %a0, %a1 {ssp.operator_type = @add1} : i32
      %1 = comb.mul %a0, %0 {ssp.operator_type = @mul2} : i32
      %2 = comb.add %a1, %a0 {ssp.operator_type = @add1} : i32
      %3 = comb.add %1, %2 {ssp.operator_type = @add1} : i32
      pipeline.return %3 : i32
    }
    hw.output %0#0 : i32
  }
}
