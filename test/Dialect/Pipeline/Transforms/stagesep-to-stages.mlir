// RUN: circt-opt -pass-pipeline='builtin.module(hw.module(pipeline.pipeline(pipeline-stagesep-to-stage)))' %s | FileCheck %s

// CHECK-LABEL:   hw.module @test1(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = pipeline.pipeline(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32, i32, i1) -> i32 {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i1):
// CHECK:             %[[VAL_9:.*]]:2, %[[VAL_10:.*]] = pipeline.stage ins %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] enable %[[VAL_8]] : (i32, i32, i1) -> (i32, i32) {
// CHECK:             ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i1, %[[VAL_14:.*]]: i1):
// CHECK:               %[[VAL_15:.*]] = comb.add %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:               pipeline.stage.return regs %[[VAL_15]], %[[VAL_11]] valid %[[VAL_14]] : (i32, i32)
// CHECK:             }
// CHECK:             %[[VAL_16:.*]]:2, %[[VAL_17:.*]] = pipeline.stage ins %[[VAL_18:.*]]#0, %[[VAL_18]]#1 enable %[[VAL_19:.*]] : (i32, i32) -> (i32, i32) {
// CHECK:             ^bb0(%[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i32, %[[VAL_22:.*]]: i1):
// CHECK:               %[[VAL_23:.*]] = comb.add %[[VAL_20]], %[[VAL_21]] : i32
// CHECK:               pipeline.stage.return regs %[[VAL_23]], %[[VAL_20]] valid %[[VAL_22]] : (i32, i32)
// CHECK:             }
// CHECK:             %[[VAL_24:.*]] = comb.add %[[VAL_25:.*]]#0, %[[VAL_25]]#1 : i32
// CHECK:             pipeline.return %[[VAL_24]] valid %[[VAL_26:.*]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_27:.*]] : i32
// CHECK:         }
hw.module @test1(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %1 = comb.add %arg0_0, %arg1_1 : i32
    %regOuts:2, %valid = pipeline.stagesep.reg enable %arg2 regs %1, %arg0_0 : i32, i32
    %2 = comb.add %regOuts#0, %regOuts#1 : i32
    %regOuts_2:2, %valid_3 = pipeline.stagesep.reg enable %valid regs %2, %regOuts#0 : i32, i32
    %3 = comb.add %regOuts_2#0, %regOuts_2#1 : i32
    pipeline.return %3 valid %valid_3 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @moveConstants(
// CHECK-SAME:             %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out: i1) {
// CHECK:           %[[VAL_3:.*]] = hw.constant true
// CHECK:           %[[VAL_4:.*]] = pipeline.pipeline(%[[VAL_0]], %[[VAL_3]]) clock %[[VAL_1]] reset %[[VAL_2]] : (i1, i1) -> i1 {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i1, %[[VAL_6:.*]]: i1):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             %[[VAL_8:.*]], %[[VAL_9:.*]] = pipeline.stage ins %[[VAL_5]], %[[VAL_6]] enable %[[VAL_6]] : (i1, i1) -> i1 {
// CHECK:             ^bb0(%[[VAL_10:.*]]: i1, %[[VAL_11:.*]]: i1, %[[VAL_12:.*]]: i1):
// CHECK:               pipeline.stage.return regs %[[VAL_10]] valid %[[VAL_12]] : (i1)
// CHECK:             }
// CHECK:             %[[VAL_13:.*]], %[[VAL_14:.*]] = pipeline.stage ins %[[VAL_15:.*]] enable %[[VAL_16:.*]] : (i1) -> i1 {
// CHECK:             ^bb0(%[[VAL_17:.*]]: i1, %[[VAL_18:.*]]: i1):
// CHECK:               %[[VAL_19:.*]] = hw.constant true
// CHECK:               %[[VAL_20:.*]] = comb.xor bin %[[VAL_19]], %[[VAL_17]] : i1
// CHECK:               pipeline.stage.return regs %[[VAL_20]] valid %[[VAL_18]] : (i1)
// CHECK:             }
// CHECK:             pipeline.return %[[VAL_21:.*]] valid %[[VAL_7]] : i1
// CHECK:           }
// CHECK:           hw.output %[[VAL_22:.*]] : i1
// CHECK:         }
hw.module @moveConstants(%ken: i1, %clk: i1, %rst: i1) -> (out: i1) {
  %true = hw.constant true
  %0 = pipeline.pipeline(%ken, %true) clock %clk reset %rst : (i1, i1) -> i1 {
  ^bb0(%arg0: i1, %arg1: i1):
    %true_0 = hw.constant true
    %outputs, %valid = pipeline.stagesep.reg enable %true_0 regs %arg0 : i1
    %1 = comb.xor bin %true_0, %outputs : i1
    %outputs_1, %valid_2 = pipeline.stagesep.reg enable %true_0 regs %1 : i1
    pipeline.return %outputs_1 valid %true_0 : i1
  }
  hw.output %0 : i1
}