// RUN: circt-opt --lower-pipeline-to-hw="outline-stages" %s | FileCheck %s

// CHECK-LABEL:   hw.module @test0_p0_s0(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, out2: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @test0_p0_s1(
// CHECK-SAME:         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32, out2: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = seq.compreg %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           hw.output %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : i32, i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @test0_p0_s2(
// CHECK-SAME:         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           hw.output %[[VAL_5]], %[[VAL_2]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @test0(
// CHECK-SAME:         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = hw.instance "test0_p0_s0" @test0_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, out2: i1)
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] = hw.instance "test0_p0_s1" @test0_p0_s1(in0: %[[VAL_5]]: i32, in1: %[[VAL_6]]: i32, in2: %[[VAL_7]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i32, out2: i1)
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = hw.instance "test0_p0_s2" @test0_p0_s2(in0: %[[VAL_8]]: i32, in1: %[[VAL_9]]: i32, in2: %[[VAL_10]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i1)
// CHECK:           hw.output %[[VAL_11]] : i32
// CHECK:         }

hw.module @test0(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %1 = comb.add %arg0_0, %arg1_1 : i32
    %regOuts:2, %valid = pipeline.stage.register when %arg2 regs %1, %arg0_0 : i32, i32
    %2 = comb.add %regOuts#0, %regOuts#1 : i32
    %regOuts_2:2, %valid_3 = pipeline.stage.register when %valid regs %2, %regOuts#0 : i32, i32
    %3 = comb.add %regOuts_2#0, %regOuts_2#1 : i32
    pipeline.return %3 valid %valid_3 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @test1_p0_s0(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           hw.output %[[VAL_0]], %[[VAL_2]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "test1_p0_s0" @test1_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i1)
// CHECK:           hw.output %[[VAL_5]] : i32
// CHECK:         }

hw.module @test1(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %1 = comb.add %arg0_0, %arg1_1 : i32
    pipeline.return %arg0_0 valid %arg2 : i32
  }
  hw.output %0 : i32
}


// CHECK-LABEL:   hw.module @test2_p0_s0(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,  %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           hw.output %[[VAL_0]], %[[VAL_2]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @test2_p1_s0(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           hw.output %[[VAL_0]], %[[VAL_2]] : i32, i1
// CHECK:         }

// CHECK-LABEL:   hw.module @test2(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = hw.instance "test2_p0_s0" @test2_p0_s0(in0: %[[VAL_0]]: i32, in1: %[[VAL_1]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i1)
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = hw.instance "test2_p1_s0" @test2_p1_s0(in0: %[[VAL_5]]: i32, in1: %[[VAL_5]]: i32, in2: %[[VAL_2]]: i1, clk: %[[VAL_3]]: i1, rst: %[[VAL_4]]: i1) -> (out0: i32, out1: i1)
// CHECK:           hw.output %[[VAL_7]] : i32
// CHECK:         }

hw.module @test2(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.pipeline(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %1 = comb.add %arg0_0, %arg1_1 : i32
    pipeline.return %arg0_0 valid %arg2 : i32
  }

  %1 = pipeline.pipeline(%0, %0, %go) clock %clk reset %rst : (i32, i32, i1) -> i32 {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %arg2: i1):
    %1 = comb.add %arg0_0, %arg1_1 : i32
    pipeline.return %arg0_0 valid %arg2 : i32
  }

  hw.output %1 : i32
}
