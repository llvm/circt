// RUN: circt-opt -pipeline-explicit-regs --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL:   hw.module @testRegsOnly(
// CHECK-SAME:            in %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLOCK:.*]] : !seq.clock, in %[[RESET:.*]] : i1, out out0 : i32, out out1 : i1) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_7:.*]] : i32 = %[[VAL_0]], %[[VAL_8:.*]] : i32 = %[[VAL_1]]) clock(%[[CLOCK]]) reset(%[[RESET]]) go(%[[GO]]) entryEn(%[[S0_VALID:.*]]) -> (out : i32) {
// CHECK:             %[[VAL_12:.*]] = comb.add %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_12]] : i32, "a0" = %[[VAL_7]] : i32)
// CHECK:           ^bb1(%[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i1):
// CHECK:             %[[VAL_16:.*]] = comb.add %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:             pipeline.stage ^bb2 regs(%[[VAL_16]] : i32, %[[VAL_13]] : i32)
// CHECK:           ^bb2(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i1):
// CHECK:             %[[VAL_20:.*]] = comb.add %[[VAL_17]], %[[VAL_18]] : i32
// CHECK:             pipeline.return %[[VAL_20]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_21:.*]], %[[VAL_22:.*]] : i32, i1
// CHECK:         }
hw.module @testRegsOnly(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out0: i32, out out1: i1) {
  %out:2 = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32){
      %add0 = comb.add %a0, %a1 : i32
      pipeline.stage ^bb1

    ^bb1(%s1_enable : i1):
      %add0_bb1 = pipeline.src %add0 : i32
      %a0_bb1 = pipeline.src %a0 : i32
      %add1 = comb.add %add0_bb1, %a0_bb1 : i32
      pipeline.stage ^bb2

    ^bb2(%s2_enable : i1):
      %add1_bb2 = pipeline.src %add1 : i32
      %add0_bb2 = pipeline.src %add0_bb1 : i32
      %add2 = comb.add %add1_bb2, %add0_bb2 : i32 // %add0 crosses multiple stages.
      pipeline.return %add2 : i32
  }
  hw.output %out#0, %out#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testLatency1(
// CHECK-SAME:          in %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLOCK:.*]] : !seq.clock, in %[[RESET:.*]] : i1, out out : i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_7:.*]] : i32 = %[[VAL_0]]) clock(%[[CLOCK]]) reset(%[[RESET]]) go(%[[GO]]) entryEn(%[[S0_VALID:.*]]) -> (out : i32) {
// CHECK:             %[[VAL_11:.*]] = hw.constant true
// CHECK:             %[[VAL_12:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_13:.*]] = comb.add %[[VAL_7]], %[[VAL_7]] : i32
// CHECK:               pipeline.latency.return %[[VAL_13]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb1  pass(%[[VAL_12]] : i32)
// CHECK:           ^bb1(%[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i1):
// CHECK:             pipeline.stage ^bb2  pass(%[[VAL_14]] : i32)
// CHECK:           ^bb2(%[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i1):
// CHECK:             pipeline.stage ^bb3 regs(%[[VAL_16]] : i32)
// CHECK:           ^bb3(%[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i1):
// CHECK:             pipeline.stage ^bb4 regs(%[[VAL_18]] : i32)
// CHECK:           ^bb4(%[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_20]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_23:.*]] : i32
// CHECK:         }
hw.module @testLatency1(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %out:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    %true = hw.constant true
    %out = pipeline.latency 2 -> (i32) {
      %d = comb.add %a0, %a0 : i32
      pipeline.latency.return %d : i32
    }
    pipeline.stage ^bb1
  ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2
  ^bb2(%s2_enable : i1):
    pipeline.stage ^bb3
  ^bb3(%s3_enable : i1):
    pipeline.stage ^bb4
  ^bb4(%s4_enable : i1):
    %out_bb4 = pipeline.src %out : i32
    pipeline.return %out_bb4 : i32
  }
  hw.output %out#0 : i32
}

// CHECK-LABEL:   hw.module @testLatency2(
// CHECK-SAME:          in %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLOCK:.*]] : !seq.clock, in %[[RESET:.*]] : i1, out out : i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_7:.*]] : i32 = %[[VAL_0]]) clock(%[[CLOCK]]) reset(%[[RESET]]) go(%[[GO]]) entryEn(%[[S0_VALID:.*]]) -> (out : i32) {
// CHECK:             %[[VAL_11:.*]] = hw.constant true
// CHECK:             %[[VAL_12:.*]] = pipeline.latency 1 -> (i32) {
// CHECK:               %[[VAL_13:.*]] = comb.add %[[VAL_7]], %[[VAL_7]] : i32
// CHECK:               pipeline.latency.return %[[VAL_13]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb1  pass(%[[VAL_14:.*]] : i32)
// CHECK:           ^bb1(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i1):
// CHECK:             pipeline.stage ^bb2 regs(%[[VAL_15]] : i32)
// CHECK:           ^bb2(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i1):
// CHECK:             %[[VAL_19:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_20:.*]] = comb.sub %[[VAL_17]], %[[VAL_17]] : i32
// CHECK:               pipeline.latency.return %[[VAL_20]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb3 regs(%[[VAL_17]] : i32) pass(%[[VAL_19]] : i32)
// CHECK:           ^bb3(%[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: i32, %[[VAL_24:.*]]: i1):
// CHECK:             pipeline.stage ^bb4 regs(%[[VAL_22]] : i32) pass(%[[VAL_23]] : i32)
// CHECK:           ^bb4(%[[VAL_25:.*]]: i32, %[[VAL_26:.*]]: i32, %[[VAL_27:.*]]: i1):
// CHECK:             %[[VAL_28:.*]] = comb.add %[[VAL_25]], %[[VAL_26]] : i32
// CHECK:             pipeline.return %[[VAL_28]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_29:.*]] : i32
// CHECK:         }

hw.module @testLatency2(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out : i32) {
  %out:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    %true = hw.constant true
    %out = pipeline.latency 1 -> (i32) {
      %d = comb.add %a0, %a0 : i32
      pipeline.latency.return %d : i32
    }
    pipeline.stage ^bb1
  ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2
  ^bb2(%s2_enable : i1):
    %out_bb2 = pipeline.src %out : i32
    %out2 = pipeline.latency 2 -> (i32) {
      %d = comb.sub %out_bb2, %out_bb2 : i32
      pipeline.latency.return %d : i32
    }
    pipeline.stage ^bb3
  ^bb3(%s3_enable : i1):
    pipeline.stage ^bb4
  ^bb4(%s4_enable : i1):
    %out_bb4 = pipeline.src %out : i32
    %out2_bb4 = pipeline.src %out2 : i32
    %res = comb.add %out_bb4, %out2_bb4 : i32
    pipeline.return %res : i32
  }
  hw.output %out#0 : i32
}

// CHECK-LABEL:   hw.module @testLatencyToLatency(
// CHECK-SAME:            in %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLOCK:.*]] : !seq.clock, in %[[RESET:.*]] : i1, out out : i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_7:.*]] : i32 = %[[VAL_0]]) clock(%[[CLOCK]]) reset(%[[RESET]]) go(%[[GO]]) entryEn(%[[S0_VALID:.*]]) -> (out : i32) {
// CHECK:             %[[VAL_11:.*]] = hw.constant true
// CHECK:             %[[VAL_12:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_13:.*]] = comb.add %[[VAL_7]], %[[VAL_7]] : i32
// CHECK:               pipeline.latency.return %[[VAL_13]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb1  pass(%[[VAL_14:.*]] : i32)
// CHECK:           ^bb1(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i1):
// CHECK:             pipeline.stage ^bb2  pass(%[[VAL_15]] : i32)
// CHECK:           ^bb2(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i1):
// CHECK:             %[[VAL_19:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_20:.*]] = hw.constant 1 : i32
// CHECK:               %[[VAL_21:.*]] = comb.add %[[VAL_17]], %[[VAL_20]] : i32
// CHECK:               pipeline.latency.return %[[VAL_21]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb3  pass(%[[VAL_22:.*]] : i32)
// CHECK:           ^bb3(%[[VAL_23:.*]]: i32, %[[VAL_24:.*]]: i1):
// CHECK:             pipeline.stage ^bb4  pass(%[[VAL_23]] : i32)
// CHECK:           ^bb4(%[[VAL_25:.*]]: i32, %[[VAL_26:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_25]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_27:.*]] : i32
// CHECK:         }
hw.module @testLatencyToLatency(in %arg0: i32, in %arg1: i32, in %go: i1, in %clk : !seq.clock, in %rst: i1, out out: i32) {
  %0:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    %true = hw.constant true
    %1 = pipeline.latency 2 -> (i32) {
      %res = comb.add %a0, %a0 : i32
      pipeline.latency.return %res : i32
    }
    pipeline.stage ^bb1
  ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2

  ^bb2(%s2_enable : i1):
    %bb2_1 = pipeline.src %1 : i32
    %2 = pipeline.latency 2 -> (i32) {
      %c1_i32 = hw.constant 1 : i32
      %res2 = comb.add %bb2_1, %c1_i32 : i32
      pipeline.latency.return %res2 : i32
    }
    pipeline.stage ^bb3

  ^bb3(%s3_enable : i1):
    pipeline.stage ^bb4

  ^bb4(%s4_enable : i1):
    %bb4_2 = pipeline.src %2 : i32
    pipeline.return %bb4_2 : i32
  }
  hw.output %0#0 : i32
}

// CHECK-LABEL:   hw.module @test_arbitrary_nesting(
// CHECK-SAME:           in %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLOCK:.*]] : !seq.clock, in %[[RESET:.*]] : i1, out out : i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = pipeline.scheduled(%[[VAL_7:.*]] : i32 = %[[VAL_0]]) clock(%[[CLOCK]]) reset(%[[RESET]]) go(%[[GO]]) entryEn(%[[S0_VALID:.*]]) -> (out : i32) {
// CHECK:             %[[VAL_11:.*]] = hw.constant true
// CHECK:             pipeline.stage ^bb1 regs("a0" = %[[VAL_7]] : i32)
// CHECK:           ^bb1(%[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i1):
// CHECK:             %[[VAL_14:.*]] = "foo.foo"(%[[VAL_12]]) : (i32) -> i32
// CHECK:             "foo.bar"() ({
// CHECK:               %[[VAL_15:.*]] = "foo.foo"(%[[VAL_12]]) : (i32) -> i32
// CHECK:               "foo.baz"() ({
// CHECK:               ^bb0(%[[VAL_16:.*]]: i32):
// CHECK:                 "foo.foobar"(%[[VAL_14]], %[[VAL_15]], %[[VAL_16]]) : (i32, i32, i32) -> ()
// CHECK:                 "foo.foobar"(%[[VAL_12]]) : (i32) -> ()
// CHECK:               }) : () -> ()
// CHECK:             }) : () -> ()
// CHECK:             pipeline.stage ^bb2 regs("a0" = %[[VAL_12]] : i32)
// CHECK:           ^bb2(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_17]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_19:.*]] : i32
// CHECK:         }
hw.module @test_arbitrary_nesting(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %out:2 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    %true = hw.constant true
    pipeline.stage ^bb1
  ^bb1(%s1_enable : i1):
    %a0_bb1 = pipeline.src %a0 : i32
    %foo = "foo.foo" (%a0_bb1) : (i32) -> (i32)
    "foo.bar" () ({
      ^bb0:
      %foo2 = "foo.foo" (%a0_bb1) : (i32) -> (i32)
      "foo.baz" () ({
        ^bb0(%innerArg0 : i32):
        // Reference all of the values defined above - none of these should
        // be registered.
        "foo.foobar" (%foo, %foo2, %innerArg0) : (i32, i32, i32) -> ()

        // Reference %a0 - this should be registered.
        "foo.foobar" (%a0_bb1) : (i32) -> ()
      }) : () -> ()
    }) : () -> ()

    pipeline.stage ^bb2
  ^bb2(%s2_enable : i1):
    %a0_bb2 = pipeline.src %a0 : i32
    pipeline.return %a0_bb2 : i32
  }
  hw.output %out#0 : i32
}

// CHECK-LABEL:   hw.module @testExtInput(
// CHECK-SAME:            in %[[VAL_0:.*]] : i32, in %[[VAL_1:.*]] : i32, in %[[GO:.*]] : i1, in %[[CLOCK:.*]] : !seq.clock, in %[[RESET:.*]] : i1, out out0 : i32, out out1 : i32) {
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = pipeline.scheduled(%[[VAL_8:.*]] : i32 = %[[VAL_0]]) clock(%[[CLOCK]]) reset(%[[RESET]]) go(%[[GO]]) entryEn(%[[S0_VALID:.*]]) -> (out0 : i32, out1 : i32) {
// CHECK:             %[[VAL_13:.*]] = hw.constant true
// CHECK:             %[[VAL_14:.*]] = comb.add %[[VAL_8]], %[[VAL_1]] : i32
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_14]] : i32)
// CHECK:           ^bb1(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_15]], %[[VAL_1]] : i32, i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_17:.*]], %[[VAL_18:.*]] : i32, i32
// CHECK:         }
hw.module @testExtInput(in %arg0 : i32, in %ext1 : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out0: i32, out out1: i32) {
  %out:3 = pipeline.scheduled(%a0 : i32 = %arg0) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out0: i32, out1: i32) {
      %true = hw.constant true
      %add0 = comb.add %a0, %ext1 : i32
      pipeline.stage ^bb1

    ^bb1(%s1_enable : i1):
      %add0_bb1 = pipeline.src %add0 : i32
      pipeline.return %add0_bb1, %ext1 : i32, i32
  }
  hw.output %out#0, %out#1 : i32, i32
}

// CHECK-LABEL:  hw.module @testNaming
// CHECK-NEXT:    %out, %done = pipeline.scheduled(%A : i32 = %myArg) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out : i32) {
// CHECK-NEXT:      %0 = pipeline.latency 2 -> (i32) {
// CHECK-NEXT:        %2 = comb.add %A, %A : i32
// CHECK-NEXT:        pipeline.latency.return %2 : i32
// CHECK-NEXT:      } {sv.namehint = "foo"}
// CHECK-NEXT:      pipeline.stage ^bb1 regs("A" = %A : i32) pass("foo" = %0 : i32)
// CHECK-NEXT:    ^bb1(%A_0: i32, %foo: i32, %s1_enable: i1):  // pred: ^bb0
// CHECK-NEXT:      pipeline.stage ^bb2 regs("A" = %A_0 : i32) pass("foo" = %foo : i32)
// CHECK-NEXT:    ^bb2(%A_1: i32, %foo_2: i32, %s2_enable: i1):  // pred: ^bb1
// CHECK-NEXT:      %1 = comb.add %A_1, %foo_2 {sv.namehint = "bar"} : i32
// CHECK-NEXT:      pipeline.stage ^bb3 regs("bar" = %1 : i32)
// CHECK-NEXT:    ^bb3(%bar: i32, %s3_enable: i1):  // pred: ^bb2
// CHECK-NEXT:      pipeline.return %bar : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    hw.output %out : i32
// CHECK-NEXT:  }
hw.module @testNaming(in %myArg : i32, in %go : i1, in %clk : !seq.clock, in %rst : i1, out out: i32) {
  %out:2 = pipeline.scheduled(%A : i32 = %myArg) clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (out: i32) {
    %res = pipeline.latency 2 -> (i32) {
      %d = comb.add %A, %A : i32
      pipeline.latency.return %d : i32
    }  {"sv.namehint" = "foo"}
    pipeline.stage ^bb1
  ^bb1(%s1_enable : i1):
    pipeline.stage ^bb2
  ^bb2(%s2_enable : i1):
    %A_bb2 = pipeline.src %A : i32
    %res_bb2 = pipeline.src %res : i32
    %0 = comb.add %A_bb2, %res_bb2  {"sv.namehint" = "bar"} : i32
    pipeline.stage ^bb3
  ^bb3(%s3_enable : i1):
    %bb3_0 = pipeline.src %0 : i32
    pipeline.return %bb3_0 : i32
  }
  hw.output %out#0 : i32
}

// CHECK-LABEL:   hw.module @pipelineLatencyCrashRepro(
// CHECK-SAME:            in %[[CLOCK:.*]] : !seq.clock, in %[[VAL_1:.*]] : i1, in %[[GO:.*]] : i1) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = pipeline.scheduled() clock(%[[CLOCK]]) reset(%[[VAL_1]]) go(%[[GO]]) entryEn(%[[S0_VALID:.*]]) -> (pipeline_done : i128) {
// CHECK:             %[[VAL_8:.*]] = "dummy.op"() : () -> i128
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_8]] : i128)
// CHECK:           ^bb1(%[[VAL_9:.*]]: i128, %[[VAL_10:.*]]: i1):
// CHECK:             %[[VAL_11:.*]] = pipeline.latency 2 -> (i54) {
// CHECK:               %[[VAL_12:.*]] = "dummy.op"() : () -> i54
// CHECK:               pipeline.latency.return %[[VAL_12]] : i54
// CHECK:             }
// CHECK:             pipeline.stage ^bb2 regs(%[[VAL_9]] : i128) pass(%[[VAL_11]] : i54)
// CHECK:           ^bb2(%[[VAL_13:.*]]: i128, %[[VAL_14:.*]]: i54, %[[VAL_15:.*]]: i1):
// CHECK:             pipeline.stage ^bb3 regs(%[[VAL_13]] : i128) pass(%[[VAL_14]] : i54)
// CHECK:           ^bb3(%[[VAL_16:.*]]: i128, %[[VAL_17:.*]]: i54, %[[VAL_18:.*]]: i1):
// CHECK:             "dummy.op"(%[[VAL_17]]) : (i54) -> ()
// CHECK:             pipeline.stage ^bb4 regs(%[[VAL_16]] : i128)
// CHECK:           ^bb4(%[[VAL_19:.*]]: i128, %[[VAL_20:.*]]: i1):
// CHECK:             pipeline.return %[[VAL_19]] : i128
// CHECK:           }
// CHECK:           hw.output
// CHECK:         }

// Tests an issue wherein the order of pass and reg operands was incorrect in
// between the order that block arguments were added to a stage, and the order
// that said block arguments were used to replace backedges within a block.

hw.module @pipelineLatencyCrashRepro(in %clk : !seq.clock, in %rst: i1, in %go: i1) {
  %pipeline_done, %done = pipeline.scheduled() clock(%clk) reset(%rst) go(%go) entryEn(%s0_enable) -> (pipeline_done : i128) {
    %0 = "dummy.op"() : () -> i128
    pipeline.stage ^bb1
  ^bb1(%s1_enable: i1):  // pred: ^bb0
    %1 = pipeline.latency 2 -> (i54) {
      %2 = "dummy.op"() : () -> i54
      pipeline.latency.return %2 : i54
    }
    pipeline.stage ^bb2
  ^bb2(%s2_enable: i1):  // pred: ^bb1
    pipeline.stage ^bb3
  ^bb3(%s3_enable: i1):  // pred: ^bb2
    %bb3_1 = pipeline.src %1 : i54
    "dummy.op"(%bb3_1) : (i54) -> ()
    pipeline.stage ^bb4
  ^bb4(%s4_enable: i1):  // pred: ^bb3
    %bb4_0 = pipeline.src %0 : i128
    pipeline.return %bb4_0 : i128
  }
  hw.output
}
