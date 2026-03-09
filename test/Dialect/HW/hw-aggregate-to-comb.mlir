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

// CHECK-LABEL: @struct_array(
hw.module private @struct_array(in %data_0 : !hw.struct<i: i2>, in %data_1 : !hw.struct<i: i2>, out data_o : !hw.array<2x!hw.struct<i: i2>>) {
  %0 = hw.array_create %data_0, %data_1 : !hw.struct<i: i2>
  // CHECK-NEXT: %[[BITCAST_1:.+]] = hw.bitcast %data_1 : (!hw.struct<i: i2>) -> i2
  // CHECK-NEXT: %[[BITCAST_0:.+]] = hw.bitcast %data_0 : (!hw.struct<i: i2>) -> i2
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[BITCAST_0]], %[[BITCAST_1]]
  // CHECK-NEXT: %[[RESULT:.+]] = hw.bitcast %[[CONCAT]] : (i4) -> !hw.array<2xstruct<i: i2>>
  // CHECK-NEXT: hw.output %[[RESULT]]
  hw.output %0 : !hw.array<2x!hw.struct<i: i2>>
}

// CHECK-LABEL: @mux_array(
hw.module private @mux_array(in %cond: i1, in %true_val: !hw.array<2xi2>, in %false_val: !hw.array<2xi2>, out out: !hw.array<2xi2>) {
  // CHECK-NEXT: %[[FALSE_BITCAST:.+]] = hw.bitcast %false_val : (!hw.array<2xi2>) -> i4
  // CHECK-NEXT: %[[TRUE_BITCAST:.+]] = hw.bitcast %true_val : (!hw.array<2xi2>) -> i4
  // CHECK-NEXT: %[[MUX:.+]] = comb.mux %cond, %[[TRUE_BITCAST]], %[[FALSE_BITCAST]] : i4
  // CHECK-NEXT: %[[RESULT:.+]] = hw.bitcast %[[MUX]] : (i4) -> !hw.array<2xi2>
  // CHECK-NEXT: hw.output %[[RESULT]]
  %0 = comb.mux %cond, %true_val, %false_val : !hw.array<2xi2>
  hw.output %0 : !hw.array<2xi2>
}

// CHECK-LABEL: @struct_extract(
hw.module private @struct_extract(in %s: !hw.struct<foo: i3, bar: i5>, out foo: i3, out bar: i5) {
  // The first field "foo" occupies the MSBs
  // struct layout: [foo (i3) | bar (i5)] = 8 bits total
  // foo is at bits [7:5] (MSB), bar is at bits [4:0] (LSB)
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %s : (!hw.struct<foo: i3, bar: i5>) -> i8
  // CHECK-NEXT: %[[FOO:.+]] = comb.extract %[[BITCAST]] from 5 : (i8) -> i3
  // CHECK-NEXT: %[[BAR:.+]] = comb.extract %[[BITCAST]] from 0 : (i8) -> i5
  // CHECK-NEXT: hw.output %[[FOO]], %[[BAR]]
  %foo = hw.struct_extract %s["foo"] : !hw.struct<foo: i3, bar: i5>
  %bar = hw.struct_extract %s["bar"] : !hw.struct<foo: i3, bar: i5>
  hw.output %foo, %bar : i3, i5
}

// CHECK-LABEL: @struct_constant_extract(
hw.module private @struct_constant_extract(out foo: i3, out bar: i5) {
  // CHECK-DAG: %[[FOO:.+]] = hw.constant 3 : i3
  // CHECK-DAG: %[[BAR:.+]] = hw.constant 5 : i5
  // CHECK-NEXT: hw.output %[[FOO]], %[[BAR]]
  %s = hw.aggregate_constant [3 : i3, 5 : i5] : !hw.struct<foo: i3, bar: i5>
  %foo = hw.struct_extract %s["foo"] : !hw.struct<foo: i3, bar: i5>
  %bar = hw.struct_extract %s["bar"] : !hw.struct<foo: i3, bar: i5>
  hw.output %foo, %bar : i3, i5
}

// CHECK-LABEL: @struct_create(
hw.module private @struct_create(in %foo: i3, in %bar: i5, out out: !hw.struct<foo: i3, bar: i5>) {
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %foo, %bar : i3, i5
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %[[CONCAT]] : (i8) -> !hw.struct<foo: i3, bar: i5>
  // CHECK-NEXT: hw.output %[[BITCAST]]
  %s = hw.struct_create (%foo, %bar) : !hw.struct<foo: i3, bar: i5>
  hw.output %s : !hw.struct<foo: i3, bar: i5>
}

// CHECK-LABEL: @struct_create_extract_roundtrip(
hw.module private @struct_create_extract_roundtrip(in %foo: i3, in %bar: i5, out foo_out: i3, out bar_out: i5) {
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %foo, %bar : i3, i5
  // CHECK-NEXT: hw.output %foo, %bar
  %s = hw.struct_create (%foo, %bar) : !hw.struct<foo: i3, bar: i5>
  %foo_out = hw.struct_extract %s["foo"] : !hw.struct<foo: i3, bar: i5>
  %bar_out = hw.struct_extract %s["bar"] : !hw.struct<foo: i3, bar: i5>
  hw.output %foo_out, %bar_out : i3, i5
}
