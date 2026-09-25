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
hw.module @array_get_for_port(in %in: !hw.array<100xi4>, out out: i4) {
  %c3_i7 = hw.constant 3 : i7
  // CHECK-NEXT: %[[BITCAST_IN:.+]] = hw.bitcast %in : (!hw.array<100xi4>) -> i400
  // CHECK-NEXT: %c3_i7 = hw.constant 3 : i7
  // CHECK-NEXT: %[[EXTRACT:.+]] = comb.extract %[[BITCAST_IN]] from 12 : (i400) -> i4
  // CHECK-NEXT: hw.output %[[EXTRACT]] : i4
  %1 = hw.array_get %in[%c3_i7] : !hw.array<100xi4>, i7
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

// CHECK-LABEL: @array_slice(
hw.module @array_slice(in %arg0: i2, in %arg1: i2, in %arg2: i2, in %arg3: i2, out out: !hw.array<4xi2>, in %sel: i2, out out_slice: !hw.array<2xi2>) {
  %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i2
  %1 = hw.array_slice %0[%sel] : (!hw.array<4xi2>) -> !hw.array<2xi2>
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %arg0, %arg1, %arg2, %arg3 : i2, i2, i2, i2
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %[[CONCAT]] : (i8) -> !hw.array<4xi2>
  // CHECK-NEXT: %[[EXTRACT_0:.+]] = comb.extract %[[CONCAT]] from 0 : (i8) -> i4
  // CHECK-NEXT: %[[EXTRACT_2:.+]] = comb.extract %[[CONCAT]] from 2 : (i8) -> i4
  // CHECK-NEXT: %[[EXTRACT_4:.+]] = comb.extract %[[CONCAT]] from 4 : (i8) -> i4
  // CHECK-NEXT: %[[EXTRACT_SEL:.+]] = comb.extract %sel from 0
  // CHECK-NEXT: %[[EXTRACT_SEL_1:.+]] = comb.extract %sel from 1
  // CHECK-NEXT: %[[MUX_0:.+]] = comb.mux %[[EXTRACT_SEL]], %[[EXTRACT_2]], %[[EXTRACT_0]]
  // CHECK-NEXT: %[[MUX_2:.+]] = comb.mux %[[EXTRACT_SEL_1]], %[[EXTRACT_4]], %[[MUX_0]]
  // CHECK-NEXT: %[[BITCAST_SLICE:.+]] = hw.bitcast %[[MUX_2]] : (i4) -> !hw.array<2xi2>
  // CHECK-NEXT: hw.output %[[BITCAST]], %[[BITCAST_SLICE]]
  hw.output %0, %1 : !hw.array<4xi2>, !hw.array<2xi2>
}

// CHECK-LABEL: @array_slice_const(
hw.module @array_slice_const(in %arg0: i2, in %arg1: i2, in %arg2: i2, in %arg3: i2, out out: !hw.array<4xi2>, out out_slice: !hw.array<2xi2>) {
  %sel = hw.constant 1 : i2
  %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i2
  %1 = hw.array_slice %0[%sel] : (!hw.array<4xi2>) -> !hw.array<2xi2>
  // CHECK-NEXT: %[[SEL:.+]] = hw.constant 1 : i2
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %arg0, %arg1, %arg2, %arg3 : i2, i2, i2, i2
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %[[CONCAT]] : (i8) -> !hw.array<4xi2>
  // CHECK-NEXT: %[[EXTRACT_2:.+]] = comb.extract %[[CONCAT]] from 2 : (i8) -> i4
  // CHECK-NEXT: %[[BITCAST_SLICE:.+]] = hw.bitcast %[[EXTRACT_2]] : (i4) -> !hw.array<2xi2>
  // CHECK-NEXT: hw.output %[[BITCAST]], %[[BITCAST_SLICE]]
  hw.output %0, %1 : !hw.array<4xi2>, !hw.array<2xi2>
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

// CHECK-LABEL: @union_create_same_width(
hw.module private @union_create_same_width(in %in: i4, out out: !hw.union<a: i4, b: i4>) {
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %in : (i4) -> !hw.union<a: i4, b: i4>
  // CHECK-NEXT: hw.output %[[BITCAST]]
  %u = hw.union_create "b", %in : !hw.union<a: i4, b: i4>
  hw.output %u : !hw.union<a: i4, b: i4>
}

// CHECK-LABEL: @union_create_padding(
hw.module private @union_create_padding(in %in: i4, out out: !hw.union<a: i8, b: i4>) {
  // CHECK-NEXT: %[[PRE:.+]] = hw.constant 0 : i4
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[PRE]], %in : i4, i4
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %[[CONCAT]] : (i8) -> !hw.union<a: i8, b: i4>
  // CHECK-NEXT: hw.output %[[BITCAST]]
  %u = hw.union_create "b", %in : !hw.union<a: i8, b: i4>
  hw.output %u : !hw.union<a: i8, b: i4>
}

// CHECK-LABEL: @union_create_offset(
hw.module private @union_create_offset(in %in: i4, out out: !hw.union<a: i8, b: i4 offset 3>) {
  // CHECK-NEXT: %[[PRE:.+]] = hw.constant false
  // CHECK-NEXT: %[[POST:.+]] = hw.constant 0 : i3
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[PRE]], %in, %[[POST]] : i1, i4, i3
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %[[CONCAT]] : (i8) -> !hw.union<a: i8, b: i4 offset 3>
  // CHECK-NEXT: hw.output %[[BITCAST]]
  %u = hw.union_create "b", %in : !hw.union<a: i8, b: i4 offset 3>
  hw.output %u : !hw.union<a: i8, b: i4 offset 3>
}

// CHECK-LABEL: @union_extract(
hw.module private @union_extract(in %u: !hw.union<a: i8, b: i4, c: i2 offset 5>, out a: i8, out b: i4, out c: i2) {
  // CHECK-NEXT: %[[BITCAST:.+]] = hw.bitcast %u : (!hw.union<a: i8, b: i4, c: i2 offset 5>) -> i8
  // CHECK-NEXT: %[[B:.+]] = comb.extract %[[BITCAST]] from 0 : (i8) -> i4
  // CHECK-NEXT: %[[C:.+]] = comb.extract %[[BITCAST]] from 5 : (i8) -> i2
  // CHECK-NEXT: hw.output %[[BITCAST]], %[[B]], %[[C]]
  %a = hw.union_extract %u["a"] : !hw.union<a: i8, b: i4, c: i2 offset 5>
  %b = hw.union_extract %u["b"] : !hw.union<a: i8, b: i4, c: i2 offset 5>
  %c = hw.union_extract %u["c"] : !hw.union<a: i8, b: i4, c: i2 offset 5>
  hw.output %a, %b, %c : i8, i4, i2
}

// CHECK-LABEL: @union_create_extract_roundtrip(
hw.module private @union_create_extract_roundtrip(in %in: i4, out a: i8, out b: i4) {
  // CHECK-NEXT: %[[PRE:.+]] = hw.constant 0 : i2
  // CHECK-NEXT: %[[POST:.+]] = hw.constant 0 : i2
  // CHECK-NEXT: %[[CAT:.+]] = comb.concat %[[PRE]], %in, %[[POST]] : i2, i4, i2
  // CHECK-NEXT: %[[B:.+]] = comb.extract %[[CAT]] from 2 : (i8) -> i4
  // CHECK-NEXT: hw.output %[[CAT]], %[[B]]
  %u = hw.union_create "b", %in : !hw.union<a: i8, b: i4 offset 2>
  %a = hw.union_extract %u["a"] : !hw.union<a: i8, b: i4 offset 2>
  %b = hw.union_extract %u["b"] : !hw.union<a: i8, b: i4 offset 2>
  hw.output %a, %b : i8, i4
}

// CHECK-LABEL: @union_bitcast(
hw.module private @union_bitcast(in %u: !hw.union<a: i8, b: i4>, out out: !hw.struct<x: i4, y: i4>) {
  // CHECK-NEXT: %[[IN:.+]] = hw.bitcast %u : (!hw.union<a: i8, b: i4>) -> i8
  // CHECK-NEXT: %[[OUT:.+]] = hw.bitcast %[[IN]] : (i8) -> !hw.struct<x: i4, y: i4>
  // CHECK-NEXT: hw.output %[[OUT]]
  %s = hw.bitcast %u : (!hw.union<a: i8, b: i4>) -> !hw.struct<x: i4, y: i4>
  hw.output %s : !hw.struct<x: i4, y: i4>
}

// CHECK-LABEL: @union_mux(
hw.module private @union_mux(in %cond: i1, in %t: !hw.union<a: i8, b: i4>, in %f: !hw.union<a: i8, b: i4>, out b: i4) {
  // CHECK-NEXT: %[[F:.+]] = hw.bitcast %f : (!hw.union<a: i8, b: i4>) -> i8
  // CHECK-NEXT: %[[T:.+]] = hw.bitcast %t : (!hw.union<a: i8, b: i4>) -> i8
  // CHECK-NEXT: %[[MUX:.+]] = comb.mux %cond, %[[T]], %[[F]] : i8
  // CHECK-NEXT: %[[B:.+]] = comb.extract %[[MUX]] from 0 : (i8) -> i4
  // CHECK-NEXT: hw.output %[[B]]
  %m = comb.mux %cond, %t, %f : !hw.union<a: i8, b: i4>
  %b = hw.union_extract %m["b"] : !hw.union<a: i8, b: i4>
  hw.output %b : i4
}

// CHECK-LABEL: @array_in_union(
hw.module private @array_in_union(in %arr: !hw.array<2xi3>, in %idx: i1, out elem: i3) {
  // CHECK-NEXT: %[[ARR:.+]] = hw.bitcast %arr : (!hw.array<2xi3>) -> i6
  // CHECK-NEXT: %[[PRE:.+]] = hw.constant false
  // CHECK-NEXT: %[[POST:.+]] = hw.constant false
  // CHECK-NEXT: %[[CAT:.+]] = comb.concat %[[PRE]], %[[ARR]], %[[POST]] : i1, i6, i1
  // CHECK-NEXT: %[[FIELD:.+]] = comb.extract %[[CAT]] from 1 : (i8) -> i6
  // CHECK-NEXT: %[[ELEM_0:.+]] = comb.extract %[[FIELD]] from 0 : (i6) -> i3
  // CHECK-NEXT: %[[ELEM_1:.+]] = comb.extract %[[FIELD]] from 3 : (i6) -> i3
  // CHECK-NEXT: %[[MUX:.+]] = comb.mux %idx, %[[ELEM_1]], %[[ELEM_0]] : i3
  // CHECK-NEXT: hw.output %[[MUX]]
  %u = hw.union_create "arr", %arr : !hw.union<raw: i8, arr: !hw.array<2xi3> offset 1>
  %a = hw.union_extract %u["arr"] : !hw.union<raw: i8, arr: !hw.array<2xi3> offset 1>
  %e = hw.array_get %a[%idx] : !hw.array<2xi3>, i1
  hw.output %e : i3
}

// CHECK-LABEL: @union_in_struct(
hw.module private @union_in_struct(in %tag: i1, in %val: i4, out tag_out: i1, out val_out: i4) {
  // CHECK-NEXT: %[[POST:.+]] = hw.constant 0 : i2
  // CHECK-NEXT: %[[CAT:.+]] = comb.concat %val, %[[POST]] : i4, i2
  // CHECK-NEXT: %[[STRUCT:.+]] = comb.concat %tag, %[[CAT]] : i1, i6
  // CHECK-NEXT: %[[VAL:.+]] = comb.extract %[[CAT]] from 2 : (i6) -> i4
  // CHECK-NEXT: hw.output %tag, %[[VAL]]
  %u = hw.union_create "b", %val : !hw.union<a: i6, b: i4 offset 2>
  %s = hw.struct_create (%tag, %u) : !hw.struct<tag: i1, data: !hw.union<a: i6, b: i4 offset 2>>
  %t = hw.struct_extract %s["tag"] : !hw.struct<tag: i1, data: !hw.union<a: i6, b: i4 offset 2>>
  %d = hw.struct_extract %s["data"] : !hw.struct<tag: i1, data: !hw.union<a: i6, b: i4 offset 2>>
  %v = hw.union_extract %d["b"] : !hw.union<a: i6, b: i4 offset 2>
  hw.output %t, %v : i1, i4
}

// CHECK-LABEL: @struct_in_union(
hw.module private @struct_in_union(in %u: !hw.union<raw: i8, s: !hw.struct<x: i2, y: i3>>, out y: i3) {
  // CHECK-NEXT: %[[CAST:.+]] = hw.bitcast %u : (!hw.union<raw: i8, s: !hw.struct<x: i2, y: i3>>) -> i8
  // CHECK-NEXT: %[[STRUCT:.+]] = comb.extract %[[CAST]] from 0 : (i8) -> i5
  // CHECK-NEXT: %[[Y:.+]] = comb.extract %[[STRUCT]] from 0 : (i5) -> i3
  // CHECK-NEXT: hw.output %[[Y]]
  %s = hw.union_extract %u["s"] : !hw.union<raw: i8, s: !hw.struct<x: i2, y: i3>>
  %y = hw.struct_extract %s["y"] : !hw.struct<x: i2, y: i3>
  hw.output %y : i3
}

// CHECK-LABEL: @union_array(
hw.module private @union_array(in %arr: !hw.array<2x!hw.union<a: i4, b: i2 offset 1>>, in %idx: i1, out b: i2) {
  // CHECK-NEXT: %[[ARR:.+]] = hw.bitcast %arr : (!hw.array<2xunion<a: i4, b: i2 offset 1>>) -> i8
  // CHECK-NEXT: %[[ELEM_0:.+]] = comb.extract %[[ARR]] from 0 : (i8) -> i4
  // CHECK-NEXT: %[[ELEM_1:.+]] = comb.extract %[[ARR]] from 4 : (i8) -> i4
  // CHECK-NEXT: %[[MUX:.+]] = comb.mux %idx, %[[ELEM_1]], %[[ELEM_0]] : i4
  // CHECK-NEXT: %[[B:.+]] = comb.extract %[[MUX]] from 1 : (i4) -> i2
  // CHECK-NEXT: hw.output %[[B]]
  %e = hw.array_get %arr[%idx] : !hw.array<2x!hw.union<a: i4, b: i2 offset 1>>, i1
  %b = hw.union_extract %e["b"] : !hw.union<a: i4, b: i2 offset 1>
  hw.output %b : i2
}
