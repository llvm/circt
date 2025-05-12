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
  // CHECK-NEXT: [[BAR_VALUE:%.*]] = sv.wire : !hw.inout<i5>
  // CHECK-NEXT: [[BAR_FOUND:%.*]] = sv.wire : !hw.inout<i1>
  // CHECK-NEXT: sv.ifdef @SYNTHESIS {
  // CHECK-NEXT:   %false = hw.constant false
  // CHECK-NEXT:   %z_i5 = sv.constantZ : i5
  // CHECK-NEXT:   sv.assign [[BAR_VALUE]], %z_i5
  // CHECK-SAME:     #sv.attribute<"This dummy assignment exists to avoid undriven lint warnings
  // CHECK-SAME:     emitAsComment
  // CHECK-NEXT:   sv.assign [[BAR_FOUND]], %false
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[FOUND_REG:%.*]] = sv.reg : !hw.inout<i32>
  // CHECK-NEXT:   [[VALUE_REG:%.*]] = sv.reg : !hw.inout<i5>
  // CHECK-NEXT:   sv.initial {
  // CHECK-NEXT:     [[BAR_STR:%.*]] = sv.constantStr "bar"
  // CHECK-NEXT:     [[PLUSARG_FOUND:%.*]] = sv.system "value$plusargs"([[BAR_STR]], [[VALUE_REG]])
  // CHECK-NEXT:     sv.bpassign [[FOUND_REG]], [[PLUSARG_FOUND]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[FOUND_READ:%.*]] = sv.read_inout [[FOUND_REG]] : !hw.inout<i32>
  // CHECK-NEXT:   [[VALUE_READ:%.*]] = sv.read_inout [[VALUE_REG]] : !hw.inout<i5>
  // CHECK-NEXT:   %c1_i32 = hw.constant 1 : i32
  // CHECK-NEXT:   [[FOUND:%.*]] = comb.icmp ceq [[FOUND_READ]], %c1_i32 : i32
  // CHECK-NEXT:   sv.assign [[BAR_FOUND]], [[FOUND]] : i1
  // CHECK-NEXT:   sv.assign [[BAR_VALUE]], [[VALUE_READ]] : i5
  // CHECK-NEXT: }
  // CHECK-NEXT: [[BAR_FOUND_READ:%.*]] = sv.read_inout [[BAR_FOUND]]
  // CHECK-NEXT: [[BAR_VALUE_READ:%.*]] = sv.read_inout [[BAR_VALUE]]
  // CHECK-NEXT: hw.output [[BAR_FOUND_READ]], [[BAR_VALUE_READ]] : i1, i5
  %0, %1 = sim.plusargs.value "bar" : i5
  hw.output %0, %1 : i1, i5
}
