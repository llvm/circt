// RUN: circt-opt %s -hw-aggregate-to-comb | FileCheck %s


// CHECK-LABEL: @agg_const
hw.module @agg_const(out out: !hw.array<4xi4>) {
  // CHECK:      %[[CONST:.+]] = hw.constant 495 : i16
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %[[CONST]] : (i16) -> !hw.array<4xi4>
  // CHECK-NEXT: hw.output %[[BITCAST]] : !hw.array<4xi4>
  %0 = hw.aggregate_constant [0 : i4, 1 : i4, -2 : i4, -1 : i4] : !hw.array<4xi4>
  hw.output %0 : !hw.array<4xi4>
}

// CHECK-LABEL: @array_get_for_port
hw.module @array_get_for_port(in %in: !hw.array<5xi4>, out out: i4) {
  %c_i2 = hw.constant 3 : i3
  // CHECK-NEXT: %[[BITCAST_IN:.+]] = hw.bitcast %in : (!hw.array<5xi4>) -> i20
  // CHECK:      %[[EXTRACT:.+]] = comb.extract %[[BITCAST_IN]] from 12 : (i20) -> i4
  // CHECK:      hw.output %[[EXTRACT]] : i4
  %1 = hw.array_get %in[%c_i2] : !hw.array<5xi4>, i3
  hw.output %1 : i4
}

// CHECK-LABEL: @array_concat
hw.module @array_concat(in %lhs: !hw.array<2xi4>, in %rhs: !hw.array<3xi4>, out out: !hw.array<5xi4>) {
  %0 = hw.array_concat %lhs, %rhs : !hw.array<2xi4>, !hw.array<3xi4>
  // CHECK-NEXT: %[[BITCAST_RHS:.+]] = hw.bitcast %rhs : (!hw.array<3xi4>) -> i12
  // CHECK-NEXT: %[[BITCAST_LHS:.+]] = hw.bitcast %lhs : (!hw.array<2xi4>) -> i8
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[BITCAST_LHS]], %[[BITCAST_RHS]] : i8, i12
  // CHECK-NEXT: %[[BITCAST_OUT:.+]] = hw.bitcast %[[CONCAT]] : (i20) -> !hw.array<5xi4>
  // CHECK:      hw.output %[[BITCAST_OUT]]
  hw.output %0 : !hw.array<5xi4>
}

hw.module.extern @foo(in %in: !hw.array<4xi2>, out out: !hw.array<4xi2>)
// CHECK-LABEL: @array_instance(
hw.module @array_instance(in %in: !hw.array<4xi2>, out out: !hw.array<4xi2>) {
  // CHECK-NEXT: hw.instance "foo" @foo(in: %in: !hw.array<4xi2>) -> (out: !hw.array<4xi2>)
  %0 = hw.instance "foo" @foo(in: %in: !hw.array<4xi2>) -> (out: !hw.array<4xi2>)
  hw.output %0 : !hw.array<4xi2>
}

// CHECK-LABEL: @array(
hw.module @array(in %arg0: i2, in %arg1: i2, in %arg2: i2, in %arg3: i2, out out: !hw.array<4xi2>, in %sel: i2, out out_get: i2) {
  %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i2
  %1 = hw.array_get %0[%sel] : !hw.array<4xi2>, i2
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %arg0, %arg1, %arg2, %arg3 : i2, i2, i2, i2
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %[[CONCAT]] : (i8) -> !hw.array<4xi2>
  // CHECK-NEXT: %[[EXTRACT_0:.+]] = comb.extract %[[CONCAT]] from 0 : (i8) -> i2
  // CHECK-NEXT: %[[EXTRACT_2:.+]] = comb.extract %[[CONCAT]] from 2 : (i8) -> i2
  // CHECK-NEXT: %[[EXTRACT_4:.+]] = comb.extract %[[CONCAT]] from 4 : (i8) -> i2
  // CHECK-NEXT: %[[EXTRACT_6:.+]] = comb.extract %[[CONCAT]] from 6 : (i8) -> i2
  // CHECK-NEXT: %[[EXTRACT_SEL:.+]] = comb.extract %sel from 0
  // CHECK-NEXT: %[[EXTRACT_SEL_1:.+]] = comb.extract %sel from 1
  // CHECK-NEXT: %[[MUX_0:.+]] = comb.mux %[[EXTRACT_SEL]], %[[EXTRACT_6]], %[[EXTRACT_4]]
  // CHECK-NEXT: %[[MUX_1:.+]] = comb.mux %[[EXTRACT_SEL]], %[[EXTRACT_2]], %[[EXTRACT_0]]
  // CHECK-NEXT: %[[MUX_2:.+]] = comb.mux %[[EXTRACT_SEL_1]], %[[MUX_0]], %[[MUX_1]]
  // CHECK-NEXT: hw.output %[[BITCAST]], %[[MUX_2]]
  hw.output %0, %1 : !hw.array<4xi2>, i2
}

// CHECK-LABEL: @array_inject(
hw.module @array_inject(in %in: !hw.array<3xi2>, in %sel: i2, in %val: i2, out out_inject: !hw.array<3xi2>) {
  // CHECK-NEXT: %[[in_bitcast:.+]] = hw.bitcast %in
  // CHECK-NEXT: %[[element_0:.+]] = comb.extract %[[in_bitcast]] from 0 : (i6) -> i2
  // CHECK-NEXT: %[[element_1:.+]] = comb.extract %[[in_bitcast]] from 2 : (i6) -> i2
  // CHECK-NEXT: %[[element_2:.+]] = comb.extract %[[in_bitcast]] from 4 : (i6) -> i2
  // CHECK-NEXT: %[[inject_2:.+]] = comb.concat %val, %[[element_1]], %[[element_0]]
  // CHECK-NEXT: %[[inject_1:.+]] = comb.concat %[[element_2]], %val, %[[element_0]]
  // CHECK-NEXT: %[[inject_0:.+]] = comb.concat %[[element_2]], %[[element_1]], %val
  // CHECK-NEXT: %[[array_2d:.+]] = comb.concat %[[inject_2]], %[[inject_1]], %[[inject_0]]
  // CHECK-NEXT: %[[array_0:.+]] = comb.extract %[[array_2d]] from 0 : (i18) -> i6
  // CHECK-NEXT: %[[array_1:.+]] = comb.extract %[[array_2d]] from 6 : (i18) -> i6
  // CHECK-NEXT: %[[array_2:.+]] = comb.extract %[[array_2d]] from 12 : (i18) -> i6
  // CHECK-NEXT: %[[sel_0:.+]] = comb.extract %sel from 0 : (i2) -> i1
  // CHECK-NEXT: %[[sel_1:.+]] = comb.extract %sel from 1 : (i2) -> i1
  // CHECK-NEXT: %[[mux_0:.+]] = comb.mux %[[sel_0]], %[[array_1]], %[[array_0]]
  // CHECK-NEXT: %[[mux_1:.+]] = comb.mux %[[sel_1]], %[[array_2]], %[[mux_0]]
  // CHECK-NEXT: %[[result:.+]] = hw.bitcast %[[mux_1]]
  // CHECK-NEXT: hw.output %[[result]]
  %0 = hw.array_inject %in[%sel], %val : !hw.array<3xi2>, i2
  hw.output %0 : !hw.array<3xi2>
}
