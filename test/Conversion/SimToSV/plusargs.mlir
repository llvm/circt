// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @plusargs_test
hw.module @plusargs_test(out test: i1) {
  // CHECK-NEXT: [[FOO_STR:%.*]] = sv.constantStr "foo"
  // CHECK-NEXT: [[FOO_DECL:%.*]] = sv.reg : !hw.inout<i1>
  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   [[TMP:%.*]] = sv.system "test$plusargs"([[FOO_STR]])
  // CHECK-NEXT:   sv.bpassign [[FOO_DECL]], [[TMP]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[FOO:%.*]] = sv.read_inout [[FOO_DECL]]
  // CHECK-NEXT: hw.output [[FOO]] : i1
  %0 = sim.plusargs.test "foo"
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @plusargs_value
hw.module @plusargs_value(out test: i1, out value: i5) {
  // CHECK-NEXT: [[BAR_VALUE_DECL:%.*]] = sv.reg : !hw.inout<i5>
  // CHECK-NEXT: [[BAR_FOUND_DECL:%.*]] = sv.reg : !hw.inout<i1>
  // CHECK-NEXT: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT:   %false = hw.constant false
  // CHECK-NEXT:   %z_i5 = sv.constantZ : i5
  // CHECK-NEXT:   sv.assign [[BAR_VALUE_DECL]], %z_i5
  // CHECK-SAME:     #sv.attribute<"This dummy assignment exists to avoid undriven lint warnings
  // CHECK-SAME:     emitAsComment
  // CHECK-NEXT:   sv.assign [[BAR_FOUND_DECL]], %false
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   sv.initial {
  // CHECK-NEXT:     %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT:     [[BAR_STR:%.*]] = sv.constantStr "bar"
  // CHECK-NEXT:     [[TMP:%.*]] = sv.system "value$plusargs"([[BAR_STR]], [[BAR_VALUE_DECL]])
  // CHECK-NEXT:     [[TMP2:%.*]] = comb.icmp bin ne [[TMP]], %c0_i32
  // CHECK-NEXT:     sv.bpassign [[BAR_FOUND_DECL]], [[TMP2]]
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: [[BAR_FOUND:%.*]] = sv.read_inout [[BAR_FOUND_DECL]]
  // CHECK-NEXT: [[BAR_VALUE:%.*]] = sv.read_inout [[BAR_VALUE_DECL]]
  // CHECK-NEXT: hw.output [[BAR_FOUND]], [[BAR_VALUE]] : i1, i5
  %0, %1 = sim.plusargs.value "bar" : i5
  hw.output %0, %1 : i1, i5
}
