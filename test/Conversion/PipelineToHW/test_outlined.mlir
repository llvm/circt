// RUN: circt-opt --lower-pipeline-to-hw="outline-stages" %s | FileCheck %s

// CHECK-LABEL:   hw.module @testSingle_p0(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "testSingle_p0_s0" @testSingle_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] = hw.instance "testSingle_p0_s1" @testSingle_p0_s1(in0: %[[VAL_5]]: i32, in1: %[[VAL_6]]: i32, enable: %[[VAL_7]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, valid: i1)
// CHECK:           %[[VAL_11:.*]] = comb.add %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_11]] : i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingle_p0_s0(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingle_p0_s1(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testSingle(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = hw.instance "testSingle_p0" @testSingle_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32)
// CHECK:           hw.output %[[VAL_5]] : i32
// CHECK:         }
hw.module @testSingle(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %outputs:2, %valid = pipeline.stage ins %arg0_0, %arg1_1 enable %arg2 : (i32, i32) -> (i32, i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2, %arg3 valid %arg5 : (i32, i32)
    }
    %outputs_2:2, %valid_3 = pipeline.stage ins %outputs#0, %outputs#1 enable %valid : (i32, i32) -> (i32, i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2, %arg3 valid %arg5 : (i32, i32)
    }
    %1 = comb.add %outputs_2#0, %outputs_2#1 : i32
    pipeline.return %1 valid %valid_3 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @testMultiple_p0(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testMultiple_p0_s0" @testMultiple_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]] : i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p0_s0(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p1(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "testMultiple_p1_s0" @testMultiple_p1_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, enable: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, valid: i1)
// CHECK:           hw.output %[[VAL_5]] : i32
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple_p1_s0(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, valid: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @testMultiple(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = hw.instance "testMultiple_p0" @testMultiple_p0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32)
// CHECK:           %[[VAL_6:.*]] = hw.instance "testMultiple_p1" @testMultiple_p1(in0: %[[VAL_5]]: i32, in1: %[[VAL_5]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32)
// CHECK:           hw.output %[[VAL_6]] : i32
// CHECK:         }
hw.module @testMultiple(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %output, %valid = pipeline.stage ins %arg0_0, %arg1_1 enable %arg2 : (i32, i32) -> (i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2 valid %arg5 : (i32)
    }
    pipeline.return %output valid %valid : i32
  }
  %1 = pipeline.pipeline(%0, %0, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %output, %valid = pipeline.stage ins %arg0_0, %arg1_1 enable %arg2 : (i32, i32) -> (i32) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %2 = comb.add %arg3, %arg4 : i32
      pipeline.stage.return regs %2 valid %arg5 : (i32)
    }
    pipeline.return %output valid %valid : i32
  }
  hw.output %1 : i32    
}

