// RUN: circt-opt --lower-pipeline-to-hw %s | FileCheck %s

// CHECK-LABEL:   hw.module @testBasic(
// CHECK-SAME:           %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1) -> (out: i1) {
// CHECK:           hw.output %[[VAL_0]] : i1
// CHECK:         }
hw.module @testBasic(%arg0: i1, %clk: i1, %rst: i1) -> (out: i1) {
  %0:2 = pipeline.scheduled(%arg0) clock %clk reset %rst go %arg0 : (i1) -> (i1) {
  ^bb0(%a0: i1, %s0_valid: i1):
    pipeline.return %a0 : i1
  }
  hw.output %0#0 : i1
}

// CHECK-LABEL:   hw.module @testLatency1(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32, done: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.add %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_6:.*]] = hw.constant false
// CHECK:           %[[VAL_7:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_6]]  : i1
// CHECK:           %[[VAL_8:.*]] = hw.constant false
// CHECK:           %[[VAL_9:.*]] = seq.compreg %[[VAL_7]], %[[VAL_3]], %[[VAL_4]], %[[VAL_8]]  : i1
// CHECK:           %[[VAL_10:.*]] = comb.mux %[[VAL_9]], %[[VAL_5]], %[[VAL_11:.*]] : i32
// CHECK:           %[[VAL_11]] = seq.compreg %[[VAL_10]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_12:.*]] = hw.constant false
// CHECK:           %[[VAL_13:.*]] = seq.compreg %[[VAL_9]], %[[VAL_3]], %[[VAL_4]], %[[VAL_12]]  : i1
// CHECK:           %[[VAL_14:.*]] = comb.mux %[[VAL_13]], %[[VAL_11]], %[[VAL_15:.*]] : i32
// CHECK:           %[[VAL_15]] = seq.compreg %[[VAL_14]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_16:.*]] = hw.constant false
// CHECK:           %[[VAL_17:.*]] = seq.compreg %[[VAL_13]], %[[VAL_3]], %[[VAL_4]], %[[VAL_16]]  : i1
// CHECK:           hw.output %[[VAL_15]], %[[VAL_17]] : i32, i1
// CHECK:         }
hw.module @testLatency1(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32, done: i1) {
  %out, %done = pipeline.scheduled(%arg0) clock %clk reset %rst go %go : (i32) -> i32 {
  ^bb0(%arg0_0: i32, %s0_valid: i1):
    %1 = pipeline.latency 2 -> (i32) {
      %6 = comb.add %arg0_0, %arg0_0 : i32
      pipeline.latency.return %6 : i32
    }
    pipeline.stage ^bb1 pass(%1 : i32)
  ^bb1(%2: i32, %s1_valid: i1):  // pred: ^bb0
    pipeline.stage ^bb2 pass(%2 : i32)
  ^bb2(%3: i32, %s2_valid: i1):  // pred: ^bb1
    pipeline.stage ^bb3 regs(%3 : i32)
  ^bb3(%4: i32, %s3_valid: i1):  // pred: ^bb2
    pipeline.stage ^bb4 regs(%4 : i32)
  ^bb4(%5: i32, %s4_valid: i1):  // pred: ^bb3
    pipeline.return %5 : i32
  }
  hw.output %out, %done : i32, i1
}

// CHECK-LABEL:   hw.module @testSingle(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           %[[VAL_12:.*]] = comb.add %[[VAL_7]], %[[VAL_9]] : i32
// CHECK:           hw.output %[[VAL_12]], %[[VAL_11]] : i32, i1
// CHECK:         }
hw.module @testSingle(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %s0_valid : i1):
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0 : i32, i32)
  ^bb1(%6: i32, %7: i32, %s1_valid : i1):  // pred: ^bb1
    %8 = comb.add %6, %7 : i32
    pipeline.return %8 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testMultiple(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = comb.sub %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = comb.mux %[[VAL_2]], %[[VAL_5]], %[[VAL_7:.*]] : i32
// CHECK:           %[[VAL_7]] = seq.compreg %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_2]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           %[[VAL_12:.*]] = comb.add %[[VAL_7]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_13:.*]] = comb.mux %[[VAL_11]], %[[VAL_12]], %[[VAL_14:.*]] : i32
// CHECK:           %[[VAL_14]] = seq.compreg %[[VAL_13]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_15:.*]] = comb.mux %[[VAL_11]], %[[VAL_7]], %[[VAL_16:.*]] : i32
// CHECK:           %[[VAL_16]] = seq.compreg %[[VAL_15]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_17:.*]] = hw.constant false
// CHECK:           %[[VAL_18:.*]] = seq.compreg %[[VAL_11]], %[[VAL_3]], %[[VAL_4]], %[[VAL_17]]  : i1
// CHECK:           %[[VAL_19:.*]] = comb.mul %[[VAL_14]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_20:.*]] = comb.sub %[[VAL_19]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_21:.*]] = comb.mux %[[VAL_2]], %[[VAL_20]], %[[VAL_22:.*]] : i32
// CHECK:           %[[VAL_22]] = seq.compreg %[[VAL_21]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_23:.*]] = comb.mux %[[VAL_2]], %[[VAL_19]], %[[VAL_24:.*]] : i32
// CHECK:           %[[VAL_24]] = seq.compreg %[[VAL_23]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_25:.*]] = hw.constant false
// CHECK:           %[[VAL_26:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_25]]  : i1
// CHECK:           %[[VAL_27:.*]] = comb.add %[[VAL_22]], %[[VAL_24]] : i32
// CHECK:           %[[VAL_28:.*]] = comb.mux %[[VAL_26]], %[[VAL_27]], %[[VAL_29:.*]] : i32
// CHECK:           %[[VAL_29]] = seq.compreg %[[VAL_28]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_30:.*]] = comb.mux %[[VAL_26]], %[[VAL_22]], %[[VAL_31:.*]] : i32
// CHECK:           %[[VAL_31]] = seq.compreg %[[VAL_30]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_32:.*]] = hw.constant false
// CHECK:           %[[VAL_33:.*]] = seq.compreg %[[VAL_26]], %[[VAL_3]], %[[VAL_4]], %[[VAL_32]]  : i1
// CHECK:           %[[VAL_34:.*]] = comb.mul %[[VAL_29]], %[[VAL_31]] : i32
// CHECK:           hw.output %[[VAL_19]], %[[VAL_18]] : i32, i1
// CHECK:         }
hw.module @testMultiple(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %s0_valid: i1):
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0 : i32, i32)
  ^bb1(%2: i32, %3: i32, %s1_valid: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5, %2 : i32, i32)
  ^bb2(%6: i32, %7: i32, %s2_valid: i1):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8 : i32
  }

  %1:2 = pipeline.scheduled(%0#0, %arg1) clock %clk reset %rst go %go : (i32, i32) -> (i32) {
  ^bb0(%arg0_0: i32, %arg1_1: i32, %s0_valid: i1):
    %1 = comb.sub %arg0_0, %arg1_1 : i32
    pipeline.stage ^bb1 regs(%1, %arg0_0 : i32, i32)
  ^bb1(%2: i32, %3: i32, %s1_valid: i1):  // pred: ^bb0
    %5 = comb.add %2, %3 : i32
    pipeline.stage ^bb2 regs(%5, %2 : i32, i32)
  ^bb2(%6: i32, %7: i32, %s2_valid: i1):  // pred: ^bb1
    %8 = comb.mul %6, %7 : i32
    pipeline.return %8 : i32
  }

  hw.output %0#0, %0#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testSingleWithExt(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.sub %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_7:.*]] = comb.mux %[[VAL_2]], %[[VAL_6]], %[[VAL_8:.*]] : i32
// CHECK:           %[[VAL_8]] = seq.compreg %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_9:.*]] = hw.constant false
// CHECK:           %[[VAL_10:.*]] = seq.compreg %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_9]]  : i1
// CHECK:           %[[VAL_11:.*]] = comb.add %[[VAL_8]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_12:.*]] = comb.mux %[[VAL_10]], %[[VAL_11]], %[[VAL_13:.*]] : i32
// CHECK:           %[[VAL_13]] = seq.compreg %[[VAL_12]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_14:.*]] = hw.constant false
// CHECK:           %[[VAL_15:.*]] = seq.compreg %[[VAL_10]], %[[VAL_3]], %[[VAL_4]], %[[VAL_14]]  : i1
// CHECK:           hw.output %[[VAL_13]], %[[VAL_1]] : i32, i32
// CHECK:         }
hw.module @testSingleWithExt(%arg0: i32, %ext1: i32, %go : i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i32) {
  %0:3 = pipeline.scheduled(%arg0, %arg0) ext (%ext1 : i32) clock %clk reset %rst go %go : (i32, i32) -> (i32, i32) {
  ^bb0(%a0: i32, %a1 : i32, %ext0: i32, %s0_valid: i1):
    %true = hw.constant true
    %1 = comb.sub %a0, %a0 : i32
    pipeline.stage ^bb1 regs(%1 : i32)

  ^bb1(%6: i32, %s1_valid: i1):
    // Use the external value inside a stage
    %8 = comb.add %6, %ext0 : i32
    pipeline.stage ^bb2 regs(%8 : i32)
  
  ^bb2(%9 : i32, %s2_valid: i1):
  // Use the external value in the exit stage.
    pipeline.return %9, %ext0  : i32, i32
  }
  hw.output %0#0, %0#1 : i32, i32
}

// CHECK-LABEL:   hw.module @testControlUsage(
// CHECK-SAME:              %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32) {
// CHECK:           %[[VAL_4:.*]] = hw.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_6:.*]] = sv.read_inout %[[VAL_5]] : !hw.inout<i32>
// CHECK:           %[[VAL_7:.*]] = comb.add %[[VAL_6]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_8:.*]] = seq.compreg.ce %[[VAL_7]], %[[VAL_2]], %[[VAL_1]], %[[VAL_3]], %[[VAL_4]]  : i32
// CHECK:           sv.assign %[[VAL_5]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_9:.*]] = comb.mux %[[VAL_1]], %[[VAL_8]], %[[VAL_10:.*]] : i32
// CHECK:           %[[VAL_10]] = seq.compreg %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_11:.*]] = hw.constant false
// CHECK:           %[[VAL_12:.*]] = seq.compreg %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_11]]  : i1
// CHECK:           %[[VAL_13:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_14:.*]] = sv.read_inout %[[VAL_13]] : !hw.inout<i32>
// CHECK:           %[[VAL_15:.*]] = comb.add %[[VAL_14]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_16:.*]] = seq.compreg.ce %[[VAL_15]], %[[VAL_2]], %[[VAL_12]], %[[VAL_3]], %[[VAL_4]]  : i32
// CHECK:           sv.assign %[[VAL_13]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_17:.*]] = comb.mux %[[VAL_12]], %[[VAL_16]], %[[VAL_18:.*]] : i32
// CHECK:           %[[VAL_18]] = seq.compreg %[[VAL_17]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_19:.*]] = hw.constant false
// CHECK:           %[[VAL_20:.*]] = seq.compreg %[[VAL_12]], %[[VAL_2]], %[[VAL_3]], %[[VAL_19]]  : i1
// CHECK:           %[[VAL_21:.*]] = sv.wire : !hw.inout<i32>
// CHECK:           %[[VAL_22:.*]] = sv.read_inout %[[VAL_21]] : !hw.inout<i32>
// CHECK:           %[[VAL_23:.*]] = comb.add %[[VAL_22]], %[[VAL_18]] : i32
// CHECK:           %[[VAL_24:.*]] = seq.compreg.ce %[[VAL_23]], %[[VAL_2]], %[[VAL_20]], %[[VAL_3]], %[[VAL_4]]  : i32
// CHECK:           sv.assign %[[VAL_21]], %[[VAL_24]] : i32
// CHECK:           hw.output %[[VAL_24]] : i32
// CHECK:         }
hw.module @testControlUsage(%arg0: i32, %go : i1, %clk: i1, %rst: i1) -> (out0: i32) {
  %0:2 = pipeline.scheduled(%arg0) ext (%clk, %rst : i1, i1) clock %clk reset %rst go %go : (i32) -> (i32) {
  ^bb0(%a0: i32, %ext_clk: i1, %ext_rst : i1, %s0_valid: i1):
    %zero = hw.constant 0 : i32
    %reg_out_wire = sv.wire : !hw.inout<i32>
    %reg_out = sv.read_inout %reg_out_wire : !hw.inout<i32>
    %add0 = comb.add %reg_out, %a0 : i32
    %out = seq.compreg.ce %add0, %ext_clk, %s0_valid, %ext_rst, %zero : i32
    sv.assign %reg_out_wire, %out : i32
    pipeline.stage ^bb1 regs(%out : i32)

  ^bb1(%6: i32, %s1_valid: i1):
    %reg1_out_wire = sv.wire : !hw.inout<i32>
    %reg1_out = sv.read_inout %reg1_out_wire : !hw.inout<i32>
    %add1 = comb.add %reg1_out, %6 : i32
    %out1 = seq.compreg.ce %add1, %ext_clk, %s1_valid, %ext_rst, %zero : i32
    sv.assign %reg1_out_wire, %out1 : i32

    pipeline.stage ^bb2 regs(%out1 : i32)
  
  ^bb2(%9 : i32, %s2_valid: i1):
    %reg2_out_wire = sv.wire : !hw.inout<i32>
    %reg2_out = sv.read_inout %reg2_out_wire : !hw.inout<i32>
    %add2 = comb.add %reg2_out, %9 : i32
    %out2 = seq.compreg.ce %add2, %ext_clk, %s2_valid, %ext_rst, %zero : i32
    sv.assign %reg2_out_wire, %out2 : i32
    pipeline.return %out2  : i32
  }
  hw.output %0#0 : i32
}

// -----

// CHECK-LABEL:   hw.module @testWithStall(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]] = hw.constant true
// CHECK:           %[[VAL_6:.*]] = comb.xor %[[VAL_2]], %[[VAL_5]] : i1
// CHECK:           %[[VAL_7:.*]] = comb.and %[[VAL_1]], %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]] = comb.mux %[[VAL_7]], %[[VAL_0]], %[[VAL_9:.*]] : i32
// CHECK:           %[[VAL_9]] = seq.compreg %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = hw.constant false
// CHECK:           %[[VAL_11:.*]] = comb.mux %[[VAL_6]], %[[VAL_1]], %[[VAL_12:.*]] : i1
// CHECK:           %[[VAL_12]] = seq.compreg %[[VAL_11]], %[[VAL_3]], %[[VAL_4]], %[[VAL_10]]  : i1
// CHECK:           hw.output %[[VAL_9]], %[[VAL_12]] : i32, i1
// CHECK:         }
hw.module @testWithStall(%arg0: i32, %go: i1, %stall : i1, %clk: i1, %rst: i1) -> (out0: i32, out1: i1) {
  %0:2 = pipeline.scheduled(%arg0) stall %stall clock %clk reset %rst go %go : (i32) -> (i32) {
  ^bb0(%arg0_0: i32, %s0_valid : i1):
    pipeline.stage ^bb1 regs(%arg0_0 : i32)
  ^bb1(%1: i32, %s1_valid : i1):  // pred: ^bb1
    pipeline.return %1 : i32
  }
  hw.output %0#0, %0#1 : i32, i1
}
