// RUN: circt-opt %s --arc-infer-state-properties | FileCheck %s

// CHECK-LABEL: arc.define @ANDBasedReset
arc.define @ANDBasedReset(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  %true = hw.constant true
  %0 = comb.xor %arg0, %true : i1
  // CHECK: [[OUT1:%.+]] = comb.or %arg1, %arg2 : i1
  %1 = comb.or %arg1, %arg2 : i1
  %2 = comb.and %0, %1 : i1
  // CHECK: arc.output [[OUT1]] : i1
  arc.output %2 : i1
}

// CHECK-LABEL: arc.define @MUXBasedReset
arc.define @MUXBasedReset(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  %false = hw.constant false
  // CHECK: [[OUT2:%.+]] = comb.or %arg1, %arg2 : i1
  %0 = comb.or %arg1, %arg2 : i1
  %1 = comb.mux %arg0, %false, %0 : i1
  // CHECK: arc.output [[OUT2]] : i1
  arc.output %1 : i1
}

// CHECK-LABEL: arc.define @enable
arc.define @enable(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> i1 {
  // CHECK-NEXT: [[V0:%.+]] = comb.or %arg1, %arg2 : i1
  %0 = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.mux %arg0, [[V0]], %arg3 : i1
  %1 = comb.mux %arg0, %0, %arg3 : i1
  // CHECK-NEXT: arc.output [[V1]] : i1
  arc.output %1 : i1
}

// CHECK-LABEL: arc.define @enableConditionHasOtherUse
arc.define @enableConditionHasOtherUse(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  // CHECK-NEXT: [[V0:%.+]] = comb.or %arg1, %arg0 : i1
  %0 = comb.or %arg1, %arg0 : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.mux %arg0, [[V0]], %arg2 : i1
  %1 = comb.mux %arg0, %0, %arg2 : i1
  // CHECK-NEXT: arc.output [[V1]] : i1
  arc.output %1 : i1
}

// CHECK-LABEL: arc.define @enableFeedbackHasOtherUse
arc.define @enableFeedbackHasOtherUse(%arg0: i1, %arg1: i1, %arg2: i1) -> i1 {
  // CHECK-NEXT: [[V0:%.+]] = comb.or %arg1, %arg2 : i1
  %0 = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.mux %arg0, [[V0]], %arg2 : i1
  %1 = comb.mux %arg0, %0, %arg2 : i1
  // CHECK-NEXT: arc.output [[V1]] : i1
  arc.output %1 : i1
}

// CHECK-LABEL: arc.define @disable
arc.define @disable(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> i1 {
  // CHECK-NEXT: [[V0:%.+]] = comb.or %arg1, %arg2 : i1
  %0 = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.mux %arg0, %arg3, [[V0]] : i1
  %1 = comb.mux %arg0, %arg3, %0 : i1
  // CHECK-NEXT: arc.output [[V1]] : i1
  arc.output %1 : i1
}

// CHECK-LABEL: arc.define @resetAndEnable
arc.define @resetAndEnable(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1) -> i1 {
  %false = hw.constant false
  // CHECK: [[V0:%.+]] = comb.or %arg1, %arg2 : i1
  %0 = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.mux %arg0, [[V0]], %arg3 : i1
  %1 = comb.mux %arg0, %0, %arg3 : i1
  %2 = comb.mux %arg4, %false, %1 : i1
  // CHECK: arc.output [[V1]] : i1
  arc.output %2 : i1
}

// CHECK-LABEL: arc.define @mixedEnableAndDisable
arc.define @mixedEnableAndDisable(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1) -> (i1, i1) {
  // CHECK-NEXT: [[OR1:%.+]] = comb.or %arg1, %arg2 : i1
  %0 = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[MUX1:%.+]] = comb.mux %arg0, %arg3, [[OR1]] : i1
  %1 = comb.mux %arg0, %arg3, %0 : i1
  // CHECK-NEXT: [[MUX2:%.+]] = comb.mux %arg0, [[OR1]], %arg4 : i1
  %2 = comb.mux %arg0, %0, %arg4 : i1
  // CHECK-NEXT: arc.output [[MUX1]], [[MUX2]] : i1, i1
  arc.output %1, %2 : i1, i1
}

// CHECK-LABEL: arc.define @mixedAndMuxReset
arc.define @mixedAndMuxReset(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1) {
  %true = hw.constant true
  %false = hw.constant false
  %0 = comb.xor %arg0, %true : i1
  // CHECK: [[OUT7:%.+]] = comb.or %arg1, %arg2 : i1
  %1 = comb.or %arg1, %arg2 : i1
  %2 = comb.and %0, %1 : i1
  %3 = comb.mux %arg0, %false, %1 : i1
  // CHECK: arc.output [[OUT7]], [[OUT7]] : i1, i1
  arc.output %2, %3 : i1, i1
}

// CHECK-LABEL: arc.define @mixedAndMuxResetDifferentConditions
arc.define @mixedAndMuxResetDifferentConditions(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> (i1, i1) {
  %true = hw.constant true
  %false = hw.constant false
  %0 = comb.xor %arg0, %true : i1
  %1 = comb.or %arg1, %arg2 : i1
  %2 = comb.and %0, %1 : i1
  %3 = comb.mux %arg3, %false, %1 : i1
  arc.output %2, %3 : i1, i1
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: [[V0:%.+]] = comb.xor %arg0, %true : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.and [[V0]], [[V1]] : i1
  // CHECK-NEXT: [[V3:%.+]] = comb.mux %arg3, %false, [[V1]] : i1
  // CHECK-NEXT: arc.output [[V2]], [[V3]] : i1, i1
}

// CHECK-LABEL: arc.define @mixedAndMuxResetLowAndHigh
arc.define @mixedAndMuxResetLowAndHigh(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1) {
  %true = hw.constant true
  %0 = comb.xor %arg0, %true : i1
  %1 = comb.or %arg1, %arg2 : i1
  %2 = comb.and %0, %1 : i1
  %3 = comb.mux %arg0, %true, %1 : i1
  arc.output %2, %3 : i1, i1
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: [[V0:%.+]] = comb.xor %arg0, %true : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.and [[V0]], [[V1]] : i1
  // CHECK-NEXT: [[V3:%.+]] = comb.mux %arg0, %true, [[V1]] : i1
  // CHECK-NEXT: arc.output [[V2]], [[V3]] : i1, i1
}

// CHECK-LABEL: arc.define @differentEnables
arc.define @differentEnables(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1, %arg4: i1) -> (i1, i1) {
  %0 = comb.or %arg1, %arg2 : i1
  %1 = comb.mux %arg0, %0, %arg3 : i1
  %2 = comb.mux %arg1, %0, %arg4 : i1
  arc.output %1, %2 : i1, i1
  // CHECK-NEXT: [[V0:%.+]] = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.mux %arg0, [[V0]], %arg3 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.mux %arg1, [[V0]], %arg4 : i1
  // CHECK-NEXT: arc.output [[V1]], [[V2]] : i1, i1
}

// CHECK-LABEL: arc.define @onlyOneEnable
arc.define @onlyOneEnable(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i1) -> (i1, i1) {
  %0 = comb.or %arg1, %arg2 : i1
  %1 = comb.mux %arg0, %arg3, %0 : i1
  arc.output %1, %0 : i1, i1
  // CHECK-NEXT: [[V0:%.+]] = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.mux %arg0, %arg3, [[V0]] : i1
  // CHECK-NEXT: arc.output [[V1]], [[V0]] : i1, i1
}

// CHECK-LABEL: arc.define @onlyOneReset
arc.define @onlyOneReset(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1) {
  %true = hw.constant true
  %false = hw.constant false
  %0 = comb.xor %arg0, %true : i1
  %1 = comb.or %arg1, %arg2 : i1
  %2 = comb.and %0, %1 : i1
  arc.output %2, %1 : i1, i1
  // CHECK-NEXT: %true = hw.constant true
  // CHECK-NEXT: %false = hw.constant false
  // CHECK-NEXT: [[V0:%.+]] = comb.xor %arg0, %true : i1
  // CHECK-NEXT: [[V1:%.+]] = comb.or %arg1, %arg2 : i1
  // CHECK-NEXT: [[V2:%.+]] = comb.and [[V0]], [[V1]] : i1
  // CHECK-NEXT: arc.output [[V2]], [[V1]] : i1, i1
}

// CHECK-LABEL: hw.module @testModule
hw.module @testModule (in %arg0: i1, in %arg1: i1, in %arg2: i1, in %arg3: i1, in %clock: !seq.clock) {
  // COM: Test: AND based reset pattern detected
  // CHECK: arc.state @ANDBasedReset(%arg0, %arg1, %arg2) clock %clock reset %arg0 latency 1 : (i1, i1, i1) -> i1
  %0 = arc.state @ANDBasedReset(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> i1

  // COM: Test: MUX based reset pattern detected
  // CHECK: arc.state @MUXBasedReset(%arg0, %arg1, %arg2) clock %clock reset %arg0 latency 1 : (i1, i1, i1) -> i1
  %1 = arc.state @MUXBasedReset(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> i1

  // COM: Test: MUX based enable pattern detected
  // CHECK: arc.state @enable(%true{{(_[0-9]+)?}}, %arg1, %arg2, %false{{(_[0-9]+)?}}) clock %clock enable %arg0 latency 1 : (i1, i1, i1, i1) -> i1
  %2 = arc.state @enable(%arg0, %arg1, %arg2, %2) clock %clock latency 1 : (i1, i1, i1, i1) -> i1

  // COM: Test: MUX based disable pattern detected
  // CHECK: [[DISABLE:%.+]] = comb.xor %arg0, %true{{(_[0-9]+)?}}
  // CHECK: arc.state @disable(%false{{(_[0-9]+)?}}, %arg1, %arg2, %false{{(_[0-9]+)?}}) clock %clock enable [[DISABLE]] latency 1 : (i1, i1, i1, i1) -> i1
  %3 = arc.state @disable(%arg0, %arg1, %arg2, %3) clock %clock latency 1 : (i1, i1, i1, i1) -> i1

  // COM: Test: both reset and enable are detected in one go
  // CHECK: arc.state @resetAndEnable(%true{{(_[0-9]+)?}}, %arg1, %arg2, %false{{(_[0-9]+)?}}, %arg3) clock %clock enable %arg0 reset %arg3 latency 1 : (i1, i1, i1, i1, i1) -> i1
  %4 = arc.state @resetAndEnable(%arg0, %arg1, %arg2, %4, %arg3) clock %clock latency 1 : (i1, i1, i1, i1, i1) -> i1

  // COM: Test: mixed enables and disables do not work
  // CHECK-NEXT: [[EN_DIS:%.+]]:2 = arc.state @mixedEnableAndDisable(%arg0, %arg1, %arg2, [[EN_DIS]]#0, [[EN_DIS]]#1) clock %clock latency 1 : (i1, i1, i1, i1, i1) -> (i1, i1)
  %5, %6 = arc.state @mixedEnableAndDisable(%arg0, %arg1, %arg2, %5, %6) clock %clock latency 1 : (i1, i1, i1, i1, i1) -> (i1, i1)

  // COM: Test: mixed MUX and AND resets for different output work
  // CHECK-NEXT: arc.state @mixedAndMuxReset(%arg0, %arg1, %arg2) clock %clock reset %arg0 latency 1 : (i1, i1, i1) -> (i1, i1)
  %7, %8 = arc.state @mixedAndMuxReset(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> (i1, i1)

  // COM: Test: mixed MUX and AND resets do not work when the reset conditions are different
  // CHECK-NEXT: arc.state @mixedAndMuxResetDifferentConditions(%arg0, %arg1, %arg2, %arg3) clock %clock latency 1 : (i1, i1, i1, i1) -> (i1, i1)
  %9, %10 = arc.state @mixedAndMuxResetDifferentConditions(%arg0, %arg1, %arg2, %arg3) clock %clock latency 1 : (i1, i1, i1, i1) -> (i1, i1)

  // COM: Test: Reset can be pulled out even if there is already a reset operand
  // CHECK-NEXT: [[NEW_RST:%.+]] = comb.or %arg1, %arg0 : i1
  // CHECK-NEXT: arc.state @MUXBasedReset(%arg0, %arg1, %arg2) clock %clock reset [[NEW_RST]] latency 1 : (i1, i1, i1) -> i1
  %11 = arc.state @MUXBasedReset(%arg0, %arg1, %arg2) clock %clock reset %arg1 latency 1 : (i1, i1, i1) -> i1

  // COM: Test: Enable can be pulled out even if there is already an enable operand
  // CHECK: [[NEW_EN:%.+]] = comb.and %arg1, %arg0 : i1
  // CHECK: arc.state @enable(%true{{(_[0-9]+)?}}, %arg1, %arg2, %false{{(_[0-9]+)?}}) clock %clock enable [[NEW_EN]] latency 1 : (i1, i1, i1, i1) -> i1
  %12 = arc.state @enable(%arg0, %arg1, %arg2, %12) clock %clock enable %arg1 latency 1 : (i1, i1, i1, i1) -> i1

  // COM: Test: Reset can be pulled out even if there is already an enable operand
  // CHECK: [[RST_COND:%.+]] = comb.and %arg1, %arg0 : i1
  // CHECK: arc.state @MUXBasedReset(%arg0, %arg1, %arg2) clock %clock enable %arg1 reset [[RST_COND]] latency 1 : (i1, i1, i1) -> i1
  %13 = arc.state @MUXBasedReset(%arg0, %arg1, %arg2) clock %clock enable %arg1 latency 1 : (i1, i1, i1) -> i1

  // COM: Test: mixed high and low resets cannot be pulled out
  // CHECK-NEXT: arc.state @mixedAndMuxResetLowAndHigh(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> (i1, i1)
  %14, %15 = arc.state @mixedAndMuxResetLowAndHigh(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> (i1, i1)

  // COM: Test: outputs with different enable conditions cannot be combined
  // CHECK-NEXT: [[EN0:%.+]]:2 = arc.state @differentEnables(%arg0, %arg1, %arg2, [[EN0]]#0, [[EN0]]#1) clock %clock latency 1 : (i1, i1, i1, i1, i1) -> (i1, i1)
  %16, %17 = arc.state @differentEnables(%arg0, %arg1, %arg2, %16, %17) clock %clock latency 1 : (i1, i1, i1, i1, i1) -> (i1, i1)

  // COM: Test: enable where not all outputs have them (some have none rather than mismatching) cannot be combined
  // CHECK-NEXT: [[EN1:%.+]]:2 = arc.state @onlyOneEnable(%arg0, %arg1, %arg2, [[EN1]]#0) clock %clock latency 1 : (i1, i1, i1, i1) -> (i1, i1)
  %18, %19 = arc.state @onlyOneEnable(%arg0, %arg1, %arg2, %18) clock %clock latency 1 : (i1, i1, i1, i1) -> (i1, i1)

  // COM: Test: reset where not all outputs have them (some have none rather than mismatching) cannot be combined
  // CHECK-NEXT: arc.state @onlyOneReset(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> (i1, i1)
  %20, %21 = arc.state @onlyOneReset(%arg0, %arg1, %arg2) clock %clock latency 1 : (i1, i1, i1) -> (i1, i1)

  // COM: the next case can in theory be supported, but requires quite a bit more complexity in the pass
  // CHECK-NEXT: [[EN2:%.+]] = arc.state @enableConditionHasOtherUse(%arg0, %arg1, [[EN2]]) clock %clock latency 1 : (i1, i1, i1) -> i1
  %22 = arc.state @enableConditionHasOtherUse(%arg0, %arg1, %22) clock %clock latency 1 : (i1, i1, i1) -> i1

  // COM: Test: When the feedback loop has another use inside the arc, we can not simply replace it with constant 0
  // CHECK: [[EN3:%.+]] = arc.state @enableFeedbackHasOtherUse(%true{{(_[0-9]+)?}}, %arg1, [[EN3]]) clock %clock enable %arg0 latency 1 : (i1, i1, i1) -> i1
  %23 = arc.state @enableFeedbackHasOtherUse(%arg0, %arg1, %23) clock %clock latency 1 : (i1, i1, i1) -> i1
}

// TODO: test that patterns handle the case where the output is used for another thing as well properly
// TODO: test that reset and enable are only added when the latency is actually 1 or higher

arc.define @arrayEnable(%enable: i1, %data: !hw.array<2xi8>, %self: !hw.array<2xi8>) -> !hw.array<2xi8> {
  %next = comb.mux %enable, %data, %self : !hw.array<2xi8>
  arc.output %next : !hw.array<2xi8>
}

arc.define @structDisable(%disable: i1, %data: !hw.struct<a: i3, b: i5>, %self: !hw.struct<a: i3, b: i5>) -> !hw.struct<a: i3, b: i5> {
  %next = comb.mux %disable, %self, %data : !hw.struct<a: i3, b: i5>
  arc.output %next : !hw.struct<a: i3, b: i5>
}

// CHECK-LABEL: hw.module @testArray
hw.module @testArray(in %clock: !seq.clock, in %enable: i1, in %data: !hw.array<2xi8>, out result: !hw.array<2xi8>) {
  // CHECK: %[[TRUE:.+]] = hw.constant true
  // CHECK: %[[ZERO:.+]] = hw.constant 0 : i16
  // CHECK: %[[FEEDBACK:.+]] = hw.bitcast %[[ZERO]] : (i16) -> !hw.array<2xi8>
  // CHECK: %[[Q:.+]] = arc.state @arrayEnable(%[[TRUE]], %data, %[[FEEDBACK]]) clock %clock enable %enable latency 1 : (i1, !hw.array<2xi8>, !hw.array<2xi8>) -> !hw.array<2xi8>
  // CHECK: hw.output %[[Q]] : !hw.array<2xi8>
  %q = arc.state @arrayEnable(%enable, %data, %q) clock %clock latency 1 : (i1, !hw.array<2xi8>, !hw.array<2xi8>) -> !hw.array<2xi8>
  hw.output %q : !hw.array<2xi8>
}

// CHECK-LABEL: hw.module @testStruct
hw.module @testStruct(in %clock: !seq.clock, in %disable: i1, in %data: !hw.struct<a: i3, b: i5>, out result: !hw.struct<a: i3, b: i5>) {
  // CHECK: %[[TRUE:.+]] = hw.constant true
  // CHECK: %[[FALSE:.+]] = hw.constant false
  // CHECK: %[[ENABLE:.+]] = comb.xor %disable, %[[TRUE]] : i1
  // CHECK: %[[ZERO:.+]] = hw.constant 0 : i8
  // CHECK: %[[FEEDBACK:.+]] = hw.bitcast %[[ZERO]] : (i8) -> !hw.struct<a: i3, b: i5>
  // CHECK: %[[Q:.+]] = arc.state @structDisable(%[[FALSE]], %data, %[[FEEDBACK]]) clock %clock enable %[[ENABLE]] latency 1 : (i1, !hw.struct<a: i3, b: i5>, !hw.struct<a: i3, b: i5>) -> !hw.struct<a: i3, b: i5>
  // CHECK: hw.output %[[Q]] : !hw.struct<a: i3, b: i5>
  %q = arc.state @structDisable(%disable, %data, %q) clock %clock latency 1 : (i1, !hw.struct<a: i3, b: i5>, !hw.struct<a: i3, b: i5>) -> !hw.struct<a: i3, b: i5>
  hw.output %q : !hw.struct<a: i3, b: i5>
}

// Float feedback has no known HW bit width. Leave the state unchanged instead
// of partially applying enable inference or attempting to create an i-1 zero.
arc.define @floatEnable(%enable: i1, %data: f32, %self: f32) -> f32 {
  %next = comb.mux %enable, %data, %self : f32
  arc.output %next : f32
}

// CHECK-LABEL: hw.module @testFloat
hw.module @testFloat(in %clock: !seq.clock, in %enable: i1, in %data: f32, out result: f32) {
  // CHECK-NEXT: %[[Q:.+]] = arc.state @floatEnable(%enable, %data, %[[Q]]) clock %clock latency 1 : (i1, f32, f32) -> f32
  // CHECK-NEXT: hw.output %[[Q]] : f32
  %q = arc.state @floatEnable(%enable, %data, %q) clock %clock latency 1 : (i1, f32, f32) -> f32
  hw.output %q : f32
}
