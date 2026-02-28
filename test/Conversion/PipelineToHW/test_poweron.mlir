// RUN: circt-opt --lower-pipeline-to-hw="enable-poweron-values" %s | FileCheck %s

// CHECK-LABEL:   hw.module @testinitial(in 
// CHECK-SAME:            %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[VAL_2:.*]] : i1, in %[[VAL_3:.*]] : !seq.clock, in %[[VAL_4:.*]] : i1, out out0 : i32, out out1 : i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg sym @p0_stage0_reg1 %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = hw.constant false
// CHECK:           %[[VAL_9:.*]] = seq.compreg sym @p0_stage1_enable %[[VAL_2]], %[[VAL_3]] reset %[[VAL_4]], %[[VAL_8]]  initial %[[INIT:.+]] : i1
// CHECK:           %[[INIT]] = seq.initial() {
// CHECK:             %false_0 = hw.constant false
// CHECK:             seq.yield %false_0 : i1
// CHECK:           } : () -> !seq.immutable<i1>
// CHECK:           %[[VAL_10:.*]] = comb.add %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           hw.output %[[VAL_10]], %[[VAL_9]] : i32, i1
// CHECK:         }

hw.module @testinitial(in %arg0: i32, in %arg1: i32, in %go: i1, in %clk: !seq.clock, in %rst: i1, out out0: i32, out out1: i1) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
    %1 = comb.sub %a0,%a1 : i32
    pipeline.stage ^bb1 regs(%1 : i32, %a0 : i32)
  ^bb1(%6: i32, %7: i32, %s1_enable : i1):  // pred: ^bb1
    %8 = comb.add %6, %7 : i32
    pipeline.return %8 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testNoReset(in 
// CHECK-SAME:                              %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[VAL_2:.*]] : i1, in %[[VAL_3:.*]] : !seq.clock, out out0 : i32, out out1 : i1) {
// CHECK:           %[[VAL_4:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:.*]] = seq.compreg sym @p0_stage0_reg0 %[[VAL_4]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg sym @p0_stage0_reg1 %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg sym @p0_stage1_enable %[[VAL_2]], %[[VAL_3]] initial %[[VAL_8:.*]] : i1
// CHECK:           %[[VAL_8]] = seq.initial() {
// CHECK:             %[[VAL_9:.*]] = hw.constant false
// CHECK:             seq.yield %[[VAL_9]] : i1
// CHECK:           } : () -> !seq.immutable<i1>
// CHECK:           %[[VAL_10:.*]] = comb.add %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           hw.output %[[VAL_10]], %[[VAL_7]] : i32, i1
// CHECK:         }
hw.module @testNoReset(in %arg0: i32, in %arg1: i32, in %go: i1, in %clk: !seq.clock, out out0: i32, out out1: i1) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) go(%go) entryEn(%s0_enable) -> (out: i32){
    %1 = comb.sub %a0,%a1 : i32
    pipeline.stage ^bb1 regs(%1 : i32, %a0 : i32)
  ^bb1(%6: i32, %7: i32, %s1_enable : i1):  // pred: ^bb1
    %8 = comb.add %6, %7 : i32
    pipeline.return %8 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}
