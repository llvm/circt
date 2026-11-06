// RUN: circt-opt --sv-check-var-drivers --split-input-file %s | FileCheck %s


// NOTE: variables are currently declared with sv.reg and driven with
// sv.assign, because the inout projection ops and the procedural assigns do
// not accept !sv.var yet. Once they do, convert these tests to sv.var and
// sv.var_assign.


// CHECK-LABEL: hw.module @SingleDriver
hw.module @SingleDriver(in %a: i8) {
  %v = sv.reg : !hw.inout<i8>
  sv.assign %v, %a : i8
}

// -----

// CHECK-LABEL: hw.module @IndependentVariables
hw.module @IndependentVariables(in %a: i8) {
  %r = sv.reg : !hw.inout<i8>
  %l = sv.logic : !hw.inout<i8>
  %v = sv.var : !sv.var<i8>
  sv.assign %r, %a : i8
  sv.assign %l, %a : i8
  sv.var_assign %v, %a : i8
}

// -----

// CHECK-LABEL: hw.module @DisjointPackedSlices
hw.module @DisjointPackedSlices(in %x: i4, in %y: i4) {
  %c0 = hw.constant 0 : i32
  %c4 = hw.constant 4 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %hi = sv.indexed_part_select_inout %bus[%c4 : 4] : !hw.inout<i8>, i32
  sv.assign %lo, %x : i4
  sv.assign %hi, %y : i4
}

// -----

// CHECK-LABEL: hw.module @DecrementingSelect
hw.module @DecrementingSelect(in %x: i4, in %y: i4) {
  %c3 = hw.constant 3 : i32
  %c4 = hw.constant 4 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c3 decrement : 4] : !hw.inout<i8>, i32
  %hi = sv.indexed_part_select_inout %bus[%c4 : 4] : !hw.inout<i8>, i32
  sv.assign %lo, %x : i4
  sv.assign %hi, %y : i4
}

// -----

// CHECK-LABEL: hw.module @DisjointStructFields
hw.module @DisjointStructFields(in %x: i8, in %y: i8) {
  %s = sv.reg : !hw.inout<struct<a: i8, b: i8>>
  %a = sv.struct_field_inout %s["a"] : !hw.inout<struct<a: i8, b: i8>>
  %b = sv.struct_field_inout %s["b"] : !hw.inout<struct<a: i8, b: i8>>
  sv.assign %a, %x : i8
  sv.assign %b, %y : i8
}

// -----

// CHECK-LABEL: hw.module @DisjointArrayElements
hw.module @DisjointArrayElements(in %x: i8, in %y: i8) {
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %e0 = sv.array_index_inout %arr[%c0] : !hw.inout<array<4xi8>>, i2
  %e1 = sv.array_index_inout %arr[%c1] : !hw.inout<array<4xi8>>, i2
  sv.assign %e0, %x : i8
  sv.assign %e1, %y : i8
}

// -----

// CHECK-LABEL: hw.module @HighBitIndices
hw.module @HighBitIndices(in %x: i8, in %y: i8) {
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %e2 = sv.array_index_inout %arr[%c2] : !hw.inout<array<4xi8>>, i2
  %e3 = sv.array_index_inout %arr[%c3] : !hw.inout<array<4xi8>>, i2
  sv.assign %e2, %x : i8
  sv.assign %e3, %y : i8
}

// -----

// CHECK-LABEL: hw.module @ArrayPartSelectCountsElements
hw.module @ArrayPartSelectCountsElements(in %x: !hw.array<2xi8>, in %y: i8, in %z: i8) {
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %sel = sv.indexed_part_select_inout %arr[%c2 : 2] : !hw.inout<array<4xi8>>, i2
  %e0 = sv.array_index_inout %arr[%c0] : !hw.inout<array<4xi8>>, i2
  %e1 = sv.array_index_inout %arr[%c1] : !hw.inout<array<4xi8>>, i2
  sv.assign %sel, %x : !hw.array<2xi8>
  sv.assign %e0, %y : i8
  sv.assign %e1, %z : i8
}

// -----

// CHECK-LABEL: hw.module @StructFieldsOfArrayElements
hw.module @StructFieldsOfArrayElements(in %x: i8, in %y: i8) {
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %arr = sv.reg : !hw.inout<array<2xstruct<a: i8, b: i8>>>
  %e0 = sv.array_index_inout %arr[%c0] : !hw.inout<array<2xstruct<a: i8, b: i8>>>, i1
  %e1 = sv.array_index_inout %arr[%c1] : !hw.inout<array<2xstruct<a: i8, b: i8>>>, i1
  %a0 = sv.struct_field_inout %e0["a"] : !hw.inout<struct<a: i8, b: i8>>
  %b1 = sv.struct_field_inout %e1["b"] : !hw.inout<struct<a: i8, b: i8>>
  sv.assign %a0, %x : i8
  sv.assign %b1, %y : i8
}

// -----

// CHECK-LABEL: hw.module @SlicesOfStructField
hw.module @SlicesOfStructField(in %x: i4, in %y: i4, in %z: i8) {
  %c0 = hw.constant 0 : i32
  %c4 = hw.constant 4 : i32
  %s = sv.reg : !hw.inout<struct<a: i8, b: i8>>
  %a = sv.struct_field_inout %s["a"] : !hw.inout<struct<a: i8, b: i8>>
  %b = sv.struct_field_inout %s["b"] : !hw.inout<struct<a: i8, b: i8>>
  %alo = sv.indexed_part_select_inout %a[%c0 : 4] : !hw.inout<i8>, i32
  %ahi = sv.indexed_part_select_inout %a[%c4 : 4] : !hw.inout<i8>, i32
  sv.assign %alo, %x : i4
  sv.assign %ahi, %y : i4
  sv.assign %b, %z : i8
}

// -----

// An i0 member drives no bit, so the driver of the whole struct does not
// conflict with it, even though the member sits inside the whole range.
// CHECK-LABEL: hw.module @ZeroWidthFieldInsideWholeDriver
hw.module @ZeroWidthFieldInsideWholeDriver(in %w: !hw.struct<a: i8, z: i0, b: i8>, in %z: i0) {
  %s = sv.reg : !hw.inout<struct<a: i8, z: i0, b: i8>>
  %zf = sv.struct_field_inout %s["z"] : !hw.inout<struct<a: i8, z: i0, b: i8>>
  sv.assign %s, %w : !hw.struct<a: i8, z: i0, b: i8>
  sv.assign %zf, %z : i0
}

// -----


// members and bits of a struct are independent
// elements, so the continuous drivers of C and of the bits of A may share the
// variable with the procedural driver of B. The drivers alternate continuous,
// procedural, continuous, and each pair is adjacent.
// CHECK-LABEL: hw.module @SpecStructExample
hw.module @SpecStructExample(in %clk: i1, in %x: i1, in %b: i8, in %c: i8) {
  %c0 = hw.constant 0 : i32
  %c1 = hw.constant 1 : i32
  %c2 = hw.constant 2 : i32
  %c3 = hw.constant 3 : i32
  %abc = sv.reg : !hw.inout<struct<A: i8, B: i8, C: i8>>
  %A = sv.struct_field_inout %abc["A"] : !hw.inout<struct<A: i8, B: i8, C: i8>>
  %B = sv.struct_field_inout %abc["B"] : !hw.inout<struct<A: i8, B: i8, C: i8>>
  %C = sv.struct_field_inout %abc["C"] : !hw.inout<struct<A: i8, B: i8, C: i8>>
  %a0 = sv.indexed_part_select_inout %A[%c0 : 1] : !hw.inout<i8>, i32
  %a1 = sv.indexed_part_select_inout %A[%c1 : 1] : !hw.inout<i8>, i32
  %a2 = sv.indexed_part_select_inout %A[%c2 : 1] : !hw.inout<i8>, i32
  %a3 = sv.indexed_part_select_inout %A[%c3 : 1] : !hw.inout<i8>, i32
  sv.assign %C, %c : i8
  sv.assign %a0, %x : i1
  sv.assign %a1, %x : i1
  sv.assign %a2, %x : i1
  sv.assign %a3, %x : i1
  sv.always posedge %clk {
    sv.passign %B, %b : i8
  }
}

// -----


// `m[1][i]` writes row 1 only, which does not contain `m[2][0]`.
// CHECK-LABEL: hw.module @DynamicIndexStaysInRow
hw.module @DynamicIndexStaysInRow(in %clk: i1, in %i: i3, in %a: i8, in %b: i8) {
  %c0 = hw.constant 0 : i3
  %c1 = hw.constant 1 : i3
  %c2 = hw.constant 2 : i3
  %m = sv.reg : !hw.inout<array<5xarray<5xi8>>>
  %row1 = sv.array_index_inout %m[%c1] : !hw.inout<array<5xarray<5xi8>>>, i3
  %row2 = sv.array_index_inout %m[%c2] : !hw.inout<array<5xarray<5xi8>>>, i3
  %dyn = sv.array_index_inout %row1[%i] : !hw.inout<array<5xi8>>, i3
  %fix = sv.array_index_inout %row2[%c0] : !hw.inout<array<5xi8>>, i3
  sv.always posedge %clk {
    sv.passign %dyn, %a : i8
  }
  sv.assign %fix, %b : i8
}

// -----

// CHECK-LABEL: hw.module @DynamicIndexStaysInField
hw.module @DynamicIndexStaysInField(in %clk: i1, in %i: i2, in %a: i8, in %b: i8) {
  %s = sv.reg : !hw.inout<struct<mem: !hw.array<4xi8>, other: i8>>
  %mem = sv.struct_field_inout %s["mem"] : !hw.inout<struct<mem: !hw.array<4xi8>, other: i8>>
  %other = sv.struct_field_inout %s["other"] : !hw.inout<struct<mem: !hw.array<4xi8>, other: i8>>
  %dyn = sv.array_index_inout %mem[%i] : !hw.inout<array<4xi8>>, i2
  sv.always posedge %clk {
    sv.passign %dyn, %a : i8
  }
  sv.assign %other, %b : i8
}
