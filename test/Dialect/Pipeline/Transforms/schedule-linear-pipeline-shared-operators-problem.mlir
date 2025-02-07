// RUN: circt-opt --pass-pipeline='builtin.module(any(pipeline-schedule-linear{problem-type="shared_operators"}))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @pipeline(
// CHECK-SAME:          in %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLOCK:.*]] : !seq.clock, in %[[RESET:.*]] : i1, out out : i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_7:.*]] : i32 = %[[VAL_0]], %[[VAL_8:.*]] : i32 = %[[VAL_1]]) clock(%[[CLOCK]]) reset(%[[RESET]]) go(%[[GO]]) entryEn(%[[VAL_9:.*]]) -> (out : i32) {
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_7]], %[[VAL_8]] {ssp.operator_type = @add1} : i32
// CHECK:             pipeline.stage ^bb1

ssp.library @lib {
  operator_type @add1 [latency<2>, limit<1>]
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

