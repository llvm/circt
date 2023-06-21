// RUN: circt-opt -pass-pipeline='builtin.module(hw.module(pipeline.scheduled(pipeline-explicit-regs)))' --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL:   hw.module @testRegsOnly(
// CHECK-SAME:         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out0: i32, out1: i1) {
// CHECK:           %[[VAL_5:.*]]:2 = pipeline.scheduled(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32, i32, i1) -> (i32, i1) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i1):
// CHECK:             %[[VAL_9:.*]] = hw.constant true
// CHECK:             %[[VAL_10:.*]] = comb.add %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_10]], %[[VAL_6]], %[[VAL_8]] : i32, i32, i1) enable %[[VAL_8]]
// CHECK:           ^bb1(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i1):
// CHECK:             %[[VAL_14:.*]] = comb.add %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:             pipeline.stage ^bb2 regs(%[[VAL_14]], %[[VAL_11]] : i32, i32) enable %[[VAL_13]]
// CHECK:           ^bb2(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32):
// CHECK:             %[[VAL_17:.*]] = comb.add %[[VAL_15]], %[[VAL_16]] : i32
// CHECK:             pipeline.return %[[VAL_17]], %[[VAL_9]] : i32, i1
// CHECK:           }
// CHECK:           hw.output %[[VAL_18:.*]]#0, %[[VAL_18]]#1 : i32, i1
// CHECK:     }        

hw.module @testRegsOnly(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out0: i32, out1: i1) {
  %out:2 = pipeline.scheduled(%arg0, %arg1, %go) clock %clk reset %rst : (i32, i32, i1) -> (i32, i1) {
    ^bb0(%a0 : i32, %a1: i32, %g : i1):
      %true = hw.constant true
      %add0 = comb.add %a0, %a1 : i32
      pipeline.stage ^bb1 enable %g
    
    ^bb1:
      %add1 = comb.add %add0, %a0 : i32 // %a0 is a block argument fed through a stage.
      pipeline.stage ^bb2 enable %g

    ^bb2:
      %add2 = comb.add %add1, %add0 : i32 // %add0 crosses multiple stages.
      pipeline.return %add2, %true : i32, i1
  }
  hw.output %out#0, %out#1 : i32, i1
}

// CHECK-LABEL:   hw.module @testLatency1(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = pipeline.scheduled(%[[VAL_0]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             %[[VAL_8:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_9:.*]] = comb.add %[[VAL_6]], %[[VAL_6]] : i32
// CHECK:               pipeline.latency.return %[[VAL_9]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb1 pass(%[[VAL_10:.*]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb1(%[[VAL_11:.*]]: i32):
// CHECK:             pipeline.stage ^bb2 pass(%[[VAL_11]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb2(%[[VAL_12:.*]]: i32):
// CHECK:             pipeline.stage ^bb3 regs(%[[VAL_12]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb3(%[[VAL_13:.*]]: i32):
// CHECK:             pipeline.stage ^bb4 regs(%[[VAL_13]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb4(%[[VAL_14:.*]]: i32):
// CHECK:             pipeline.return %[[VAL_14]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_15:.*]] : i32
// CHECK:         }
hw.module @testLatency1(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %out = pipeline.scheduled(%arg0) clock %clk reset %rst : (i32) -> (i32) {
  ^bb0(%a0 : i32):
    %true = hw.constant true
    %out = pipeline.latency 2 -> (i32) {
      %r = comb.add %a0, %a0 : i32
      pipeline.latency.return %r : i32
    }
    pipeline.stage ^bb1 enable %true
  ^bb1:
    pipeline.stage ^bb2 enable %true
  ^bb2:
    pipeline.stage ^bb3 enable %true
  ^bb3:
    pipeline.stage ^bb4 enable %true
  ^bb4:
    pipeline.return %out : i32
  }
  hw.output %out : i32
}

// CHECK-LABEL:   hw.module @testLatency2(
// CHECK-SAME:              %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = pipeline.scheduled(%[[VAL_0]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             %[[VAL_8:.*]] = pipeline.latency 1 -> (i32) {
// CHECK:               %[[VAL_9:.*]] = comb.add %[[VAL_6]], %[[VAL_6]] : i32
// CHECK:               pipeline.latency.return %[[VAL_9]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb1 pass(%[[VAL_10:.*]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb1(%[[VAL_11:.*]]: i32):
// CHECK:             pipeline.stage ^bb2 regs(%[[VAL_11]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb2(%[[VAL_12:.*]]: i32):
// CHECK:             %[[VAL_13:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_14:.*]] = comb.sub %[[VAL_12]], %[[VAL_12]] : i32
// CHECK:               pipeline.latency.return %[[VAL_14]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb3 regs(%[[VAL_12]] : i32) pass(%[[VAL_15:.*]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb3(%[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i32):
// CHECK:             pipeline.stage ^bb4 regs(%[[VAL_16]] : i32) pass(%[[VAL_17]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb4(%[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i32):
// CHECK:             %[[VAL_20:.*]] = comb.add %[[VAL_18]], %[[VAL_19]] : i32
// CHECK:             pipeline.return %[[VAL_18]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_21:.*]] : i32
// CHECK:         }
hw.module @testLatency2(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %out = pipeline.scheduled(%arg0) clock %clk reset %rst : (i32) -> (i32) {
  ^bb0(%a0 : i32):
    %true = hw.constant true
    %out = pipeline.latency 1 -> (i32) {
      %r = comb.add %a0, %a0 : i32
      pipeline.latency.return %r : i32
    }
    pipeline.stage ^bb1 enable %true
  ^bb1:
    pipeline.stage ^bb2 enable %true
  ^bb2:
    %out2 = pipeline.latency 2 -> (i32) {
      %r = comb.sub %out, %out : i32
      pipeline.latency.return %r : i32
    }
    pipeline.stage ^bb3 enable %true
  ^bb3:
    pipeline.stage ^bb4 enable %true
  ^bb4:
    %res = comb.add %out, %out2 : i32
    pipeline.return %out : i32
  }
  hw.output %out : i32
}

// CHECK-LABEL:   hw.module @testLatencyToLatency(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = pipeline.scheduled(%[[VAL_0]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             %[[OUT1:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_9:.*]] = comb.add %[[VAL_6]], %[[VAL_6]] : i32
// CHECK:               pipeline.latency.return %[[VAL_9]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb1 pass(%[[OUT1]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb1(%[[PASS1:.*]]: i32):
// CHECK:             pipeline.stage ^bb2 pass(%[[PASS1]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb2(%[[PASS2:.*]]: i32):
// CHECK:             %[[VAL_13:.*]] = pipeline.latency 2 -> (i32) {
// CHECK:               %[[VAL_14:.*]] = hw.constant 1 : i32
// CHECK:               %[[VAL_15:.*]] = comb.add %[[PASS2]], %[[VAL_14]] : i32
// CHECK:               pipeline.latency.return %[[VAL_15]] : i32
// CHECK:             }
// CHECK:             pipeline.stage ^bb3 pass(%[[VAL_16:.*]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb3(%[[VAL_17:.*]]: i32):
// CHECK:             pipeline.stage ^bb4 pass(%[[VAL_17]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb4(%[[VAL_18:.*]]: i32):
// CHECK:             pipeline.return %[[VAL_18]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_19:.*]] : i32
// CHECK:         }
hw.module @testLatencyToLatency(%arg0: i32, %arg1: i32, %go: i1, %clk: i1, %rst: i1) -> (out: i32) {
  %0 = pipeline.scheduled(%arg0) clock %clk reset %rst : (i32) -> i32 {
  ^bb0(%arg0_0: i32):
    %true = hw.constant true
    %1 = pipeline.latency 2 -> (i32) {
      %res = comb.add %arg0_0, %arg0_0 : i32
      pipeline.latency.return %res : i32
    }
    pipeline.stage ^bb1 enable %true
  ^bb1:
    pipeline.stage ^bb2 enable %true

  ^bb2:
    %2 = pipeline.latency 2 -> (i32) {
      %c1_i32 = hw.constant 1 : i32
      %res2 = comb.add %1, %c1_i32 : i32
      pipeline.latency.return %res2 : i32
    }
    pipeline.stage ^bb3 enable %true

  ^bb3:
    pipeline.stage ^bb4 enable %true

  ^bb4:
    pipeline.return %2 : i32
  }
  hw.output %0 : i32
}

// CHECK-LABEL:   hw.module @test_arbitrary_nesting(
// CHECK-SAME:          %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: i1) -> (out: i32) {
// CHECK:           %[[VAL_5:.*]] = pipeline.scheduled(%[[VAL_0]]) clock %[[VAL_3]] reset %[[VAL_4]] : (i32) -> i32 {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_6]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb1(%[[VAL_8:.*]]: i32):
// CHECK:             %[[VAL_9:.*]] = "foo.foo"(%[[VAL_8]]) : (i32) -> i32
// CHECK:             "foo.bar"() ({
// CHECK:               %[[VAL_10:.*]] = "foo.foo"(%[[VAL_8]]) : (i32) -> i32
// CHECK:               "foo.baz"() ({
// CHECK:               ^bb0(%[[VAL_11:.*]]: i32):
// CHECK:                 "foo.foobar"(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) : (i32, i32, i32) -> ()
// CHECK:                 "foo.foobar"(%[[VAL_8]]) : (i32) -> ()
// CHECK:               }) : () -> ()
// CHECK:             }) : () -> ()
// CHECK:             pipeline.stage ^bb2 regs(%[[VAL_8]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb2(%[[VAL_12:.*]]: i32):
// CHECK:             pipeline.return %[[VAL_12]] : i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_13:.*]] : i32
// CHECK:         }
hw.module @test_arbitrary_nesting(%arg0 : i32, %arg1 : i32, %go : i1, %clk : i1, %rst : i1) -> (out: i32) {
  %out = pipeline.scheduled(%arg0) clock %clk reset %rst : (i32) -> (i32) {
  ^bb0(%a0 : i32):
    %true = hw.constant true
    pipeline.stage ^bb1 enable %true
  ^bb1:
    %foo = "foo.foo" (%a0) : (i32) -> (i32)
    "foo.bar" () ({
      ^bb0:
      %foo2 = "foo.foo" (%a0) : (i32) -> (i32)
      "foo.baz" () ({
        ^bb0(%innerArg0 : i32):
        // Reference all of the values defined above - none of these should
        // be registered.
        "foo.foobar" (%foo, %foo2, %innerArg0) : (i32, i32, i32) -> ()

        // Reference %a0 - this should be registered.
        "foo.foobar" (%a0) : (i32) -> ()
      }) : () -> ()
    }) : () -> ()

    pipeline.stage ^bb2 enable %true
  ^bb2:
    pipeline.return %a0 : i32
  }
  hw.output %out : i32
}

// CHECK-LABEL:   hw.module @testExtInput(
// CHECK-SAME:            %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: i1) -> (out0: i32, out1: i32) {
// CHECK:           %[[VAL_4:.*]]:2 = pipeline.scheduled(%[[VAL_0]]) ext(%[[VAL_1]] : i32) clock %[[VAL_2]] reset %[[VAL_3]] : (i32) -> (i32, i32) {
// CHECK:           ^bb0(%[[ARG_IN:.*]]: i32, %[[EXT_IN:.*]]: i32):
// CHECK:             %[[VAL_7:.*]] = hw.constant true
// CHECK:             %[[VAL_8:.*]] = comb.add %[[ARG_IN]], %[[EXT_IN]] : i32
// CHECK:             pipeline.stage ^bb1 regs(%[[VAL_8]] : i32) enable %[[VAL_7]]
// CHECK:           ^bb1(%[[VAL_9:.*]]: i32):
// CHECK:             pipeline.return %[[VAL_9]], %[[EXT_IN]] : i32, i32
// CHECK:           }
// CHECK:           hw.output %[[VAL_10:.*]]#0, %[[VAL_10]]#1 : i32, i32
// CHECK:         }
hw.module @testExtInput(%arg0 : i32, %ext1 : i32, %clk : i1, %rst : i1) -> (out0: i32, out1: i32) {
  %out:2 = pipeline.scheduled(%arg0) ext(%ext1 : i32) clock %clk reset %rst : (i32) -> (i32, i32) {
    ^bb0(%a0 : i32, %e0: i32):
      %true = hw.constant true
      %add0 = comb.add %a0, %e0 : i32
      pipeline.stage ^bb1 enable %true

    ^bb1:
      pipeline.return %add0, %e0 : i32, i32
  }
  hw.output %out#0, %out#1 : i32, i32
}
