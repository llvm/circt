// RUN: circt-opt -pass-pipeline='builtin.module(hw.module(pipeline.pipeline(pipeline-ss-to-stage)))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @test1(
// CHECK-SAME:        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1,
// CHECK-SAME:                     %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = pipeline.pipeline(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32, i32, i1) -> i32 {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i1):
// CHECK:             %[[VAL_9:.*]]:2, %[[VAL_10:.*]] = pipeline.stage ins %[[VAL_6]], %[[VAL_7]] enable %[[VAL_8]] : (i32, i32) -> (i32, i32) {
// CHECK:             ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i1):
// CHECK:               %[[VAL_14:.*]] = comb.add %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:               pipeline.stage.return regs %[[VAL_14]], %[[VAL_11]] valid %[[VAL_13]] : (i32, i32)
// CHECK:             }
// CHECK:             %[[VAL_15:.*]]:2, %[[VAL_16:.*]] = pipeline.stage ins %[[VAL_17:.*]]#0, %[[VAL_17]]#1 enable %[[VAL_18:.*]] : (i32, i32) -> (i32, i32) {
// CHECK:             ^bb0(%[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i1):
// CHECK:               %[[VAL_22:.*]] = comb.add %[[VAL_19]], %[[VAL_20]] : i32
// CHECK:               pipeline.stage.return regs %[[VAL_22]], %[[VAL_19]] valid %[[VAL_21]] : (i32, i32)
// CHECK:             }
// CHECK:             %[[VAL_23:.*]] = comb.add %[[VAL_24:.*]]#0, %[[VAL_24]]#1 : i32
// CHECK:             pipeline.return %[[VAL_23]] valid %[[VAL_25:.*]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_26:.*]] : i32
// CHECK:         }
hw.module @test1(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %1 = comb.add %arg0_0, %arg1_1 : i32
    %regOuts:2, %valid = pipeline.ss.reg enable %arg2 regs %1, %arg0_0 : i32, i32
    %2 = comb.add %regOuts#0, %regOuts#1 : i32
    %regOuts_2:2, %valid_3 = pipeline.ss.reg enable %valid regs %2, %regOuts#0 : i32, i32
    %3 = comb.add %regOuts_2#0, %regOuts_2#1 : i32
    pipeline.return %3 valid %valid_3 : i32
  }
  hw.output %0 : i32
}
