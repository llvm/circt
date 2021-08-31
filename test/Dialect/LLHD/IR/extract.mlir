// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt | circt-opt | FileCheck %s

// CHECK-LABEL: @extract_slice_integers
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI32:.*]]: i32
func @extract_slice_integers(%cI1 : i1, %cI32 : i32) {
  // CHECK-NEXT: %{{.*}} = llhd.extract_slice %[[CI1]], 0 : i1 -> i1
  %0 = llhd.extract_slice %cI1, 0 : i1 -> i1
  // CHECK-NEXT: %{{.*}} = llhd.extract_slice %[[CI32]], 0 : i32 -> i5
  %1 = llhd.extract_slice %cI32, 0 : i32 -> i5

  return
}

// CHECK-LABEL: @extract_slice_signals
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI32:.*]]: !llhd.sig<i32>
func @extract_slice_signals (%sI1 : !llhd.sig<i1>, %sI32 : !llhd.sig<i32>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.extract_slice %[[SI1]], 0 : !llhd.sig<i1> -> !llhd.sig<i1>
  %0 = llhd.extract_slice %sI1, 0 : !llhd.sig<i1> -> !llhd.sig<i1>
  // CHECK-NEXT: %{{.*}} = llhd.extract_slice %[[SI32]], 0 : !llhd.sig<i32> -> !llhd.sig<i5>
  %1 = llhd.extract_slice %sI32, 0 : !llhd.sig<i32> -> !llhd.sig<i5>

  return
}

// CHECK-LABEL: @extract_slice_array_signals
// CHECK-SAME: %[[ARRAY5:.*]]: !llhd.sig<!hw.array<5xi1>>
// CHECK-SAME: %[[ARRAY1:.*]]: !llhd.sig<!hw.array<1xi32>>
func @extract_slice_array_signals (%array5 : !llhd.sig<!hw.array<5xi1>>, %array1 : !llhd.sig<!hw.array<1xi32>>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.extract_slice %[[ARRAY5]], 2 : !llhd.sig<!hw.array<5xi1>> -> !llhd.sig<!hw.array<3xi1>>
  %0 = llhd.extract_slice %array5, 2 : !llhd.sig<!hw.array<5xi1>> -> !llhd.sig<!hw.array<3xi1>>
  // CHECK-NEXT: %{{.*}} = llhd.extract_slice %[[ARRAY1]], 0 : !llhd.sig<!hw.array<1xi32>> -> !llhd.sig<!hw.array<1xi32>>
  %1 = llhd.extract_slice %array1, 0 : !llhd.sig<!hw.array<1xi32>> -> !llhd.sig<!hw.array<1xi32>>

  return
}

// CHECK-LABEL: @dyn_extract_slice_integers
// CHECK-SAME: %[[CI1:.*]]: i1,
// CHECK-SAME: %[[CI32:.*]]: i32,
// CHECK-SAME: %[[IND0:.*]]: i5,
// CHECK-SAME: %[[IND1:.*]]: i10
func @dyn_extract_slice_integers(%cI1 : i1, %cI32 : i32, %i0 : i5, %i1 : i10) {
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[CI1]], %[[IND0]] : (i1, i5) -> i1
  %0 = llhd.dyn_extract_slice %cI1, %i0 : (i1, i5) -> i1
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[CI32]], %[[IND1]] : (i32, i10) -> i15
  %1 = llhd.dyn_extract_slice %cI32, %i1 : (i32, i10) -> i15

  return
}

// CHECK-LABEL: @dyn_extract_slice_signals
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>,
// CHECK-SAME: %[[SI32:.*]]: !llhd.sig<i32>,
// CHECK-SAME: %[[IND0:.*]]: i5,
// CHECK-SAME: %[[IND1:.*]]: i10
func @dyn_extract_slice_signals (%sI1 : !llhd.sig<i1>, %sI32 : !llhd.sig<i32>, %i0 : i5, %i1 : i10) {
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[SI1]], %[[IND0]] : (!llhd.sig<i1>, i5) -> !llhd.sig<i1>
  %0 = llhd.dyn_extract_slice %sI1, %i0 : (!llhd.sig<i1>, i5) -> !llhd.sig<i1>
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[SI32]], %[[IND1]] : (!llhd.sig<i32>, i10) -> !llhd.sig<i5>
  %1 = llhd.dyn_extract_slice %sI32, %i1 : (!llhd.sig<i32>, i10) -> !llhd.sig<i5>

  return
}

// CHECK-LABEL: @dyn_extract_slice_array_signals
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<!hw.array<5xi1>>,
// CHECK-SAME: %[[SI32:.*]]: !llhd.sig<!hw.array<1xi32>>,
// CHECK-SAME: %[[IND0:.*]]: i5,
// CHECK-SAME: %[[IND1:.*]]: i10
func @dyn_extract_slice_array_signals (%sI1 : !llhd.sig<!hw.array<5xi1>>, %sI32 : !llhd.sig<!hw.array<1xi32>>, %i0 : i5, %i1 : i10) {
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[SI1]], %[[IND0]] : (!llhd.sig<!hw.array<5xi1>>, i5) -> !llhd.sig<!hw.array<2xi1>>
  %0 = llhd.dyn_extract_slice %sI1, %i0 : (!llhd.sig<!hw.array<5xi1>>, i5) -> !llhd.sig<!hw.array<2xi1>>
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[SI32]], %[[IND1]] : (!llhd.sig<!hw.array<1xi32>>, i10) -> !llhd.sig<!hw.array<1xi32>>
  %1 = llhd.dyn_extract_slice %sI32, %i1 : (!llhd.sig<!hw.array<1xi32>>, i10) -> !llhd.sig<!hw.array<1xi32>>

  return
}

// CHECK-LABEL: @dyn_extract_slice_vec
// CHECK-SAME: %[[V1:.*]]: !hw.array<1xi1>,
// CHECK-SAME: %[[V10:.*]]: !hw.array<10xi1>,
// CHECK-SAME: %[[IND0:.*]]: i5,
// CHECK-SAME: %[[IND1:.*]]: i10
func @dyn_extract_slice_vec(%v1 : !hw.array<1xi1>, %v10 : !hw.array<10xi1>, %i0 : i5, %i1 : i10) {
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[V1]], %[[IND0]] : (!hw.array<1xi1>, i5) -> !hw.array<1xi1>
  %0 = llhd.dyn_extract_slice %v1, %i0 : (!hw.array<1xi1>, i5) -> !hw.array<1xi1>
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_slice %[[V10]], %[[IND1]] : (!hw.array<10xi1>, i10) -> !hw.array<5xi1>
  %1 = llhd.dyn_extract_slice %v10, %i1 : (!hw.array<10xi1>, i10) -> !hw.array<5xi1>

  return
}

// CHECK-LABEL: @extract_element_arrays
// CHECK-SAME: %[[ARRAY1:.*]]: !hw.array<1xi1>
// CHECK-SAME: %[[ARRAY5:.*]]: !hw.array<5xi32>
func @extract_element_arrays(%array1 : !hw.array<1xi1>, %array5 : !hw.array<5xi32>) {
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[ARRAY1]], 0 : !hw.array<1xi1> -> i1
  %0 = llhd.extract_element %array1, 0 : !hw.array<1xi1> -> i1
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[ARRAY5]], 4 : !hw.array<5xi32> -> i32
  %1 = llhd.extract_element %array5, 4 : !hw.array<5xi32> -> i32

  return
}

// CHECK-LABEL: @extract_element_tuples
// CHECK-SAME: %[[TUP:.*]]: !hw.struct<foo: i1, bar: i2, baz: i3>
func @extract_element_tuples(%tup : !hw.struct<foo: i1, bar: i2, baz: i3>) {
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[TUP]], 0 : !hw.struct<foo: i1, bar: i2, baz: i3> -> i1
  %0 = llhd.extract_element %tup, 0 : !hw.struct<foo: i1, bar: i2, baz: i3> -> i1
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[TUP]], 2 : !hw.struct<foo: i1, bar: i2, baz: i3> -> i3
  %1 = llhd.extract_element %tup, 2 : !hw.struct<foo: i1, bar: i2, baz: i3> -> i3

  return
}

// CHECK-LABEL: @extract_element_signals_of_arrays
// CHECK-SAME: %[[ARRAY1:.*]]: !llhd.sig<!hw.array<1xi1>>
// CHECK-SAME: %[[ARRAY5:.*]]: !llhd.sig<!hw.array<5xi32>>
func @extract_element_signals_of_arrays(%array1 : !llhd.sig<!hw.array<1xi1>>, %array5 : !llhd.sig<!hw.array<5xi32>>) {
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[ARRAY1]], 0 : !llhd.sig<!hw.array<1xi1>> -> !llhd.sig<i1>
  %0 = llhd.extract_element %array1, 0 : !llhd.sig<!hw.array<1xi1>> -> !llhd.sig<i1>
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[ARRAY5]], 4 : !llhd.sig<!hw.array<5xi32>> -> !llhd.sig<i32>
  %1 = llhd.extract_element %array5, 4 : !llhd.sig<!hw.array<5xi32>> -> !llhd.sig<i32>

  return
}

// CHECK-LABEL: @extract_element_signals_of_tuples
// CHECK-SAME: %[[TUP:.*]]: !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>
func @extract_element_signals_of_tuples(%tup : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[TUP]], 0 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>> -> !llhd.sig<i1>
  %0 = llhd.extract_element %tup, 0 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>> -> !llhd.sig<i1>
  // CHECK-NEXT: %{{.*}} = llhd.extract_element %[[TUP]], 2 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>> -> !llhd.sig<i3>
  %1 = llhd.extract_element %tup, 2 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>> -> !llhd.sig<i3>

  return
}

// CHECK-LABEL: @dyn_extract_element_arrays
// CHECK-SAME: %[[INDEX:.*]]: i32
// CHECK-SAME: %[[ARRAY1:.*]]: !hw.array<1xi1>
// CHECK-SAME: %[[ARRAY5:.*]]: !hw.array<5xi32>
func @dyn_extract_element_arrays(%index : i32, %array1 : !hw.array<1xi1>, %array5 : !hw.array<5xi32>) {
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_element %[[ARRAY1]], %[[INDEX]] : (!hw.array<1xi1>, i32) -> i1
  %0 = llhd.dyn_extract_element %array1, %index : (!hw.array<1xi1>, i32) -> i1
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_element %[[ARRAY5]], %[[INDEX]] : (!hw.array<5xi32>, i32) -> i32
  %1 = llhd.dyn_extract_element %array5, %index : (!hw.array<5xi32>, i32) -> i32

  return
}

// CHECK-LABEL: @dyn_extract_element_signals_of_arrays
// CHECK-SAME: %[[INDEX:.*]]: i32
// CHECK-SAME: %[[ARRAY1:.*]]: !llhd.sig<!hw.array<1xi1>>
// CHECK-SAME: %[[ARRAY5:.*]]: !llhd.sig<!hw.array<5xi32>>
func @dyn_extract_element_signals_of_arrays(%index : i32, %array1 : !llhd.sig<!hw.array<1xi1>>, %array5 : !llhd.sig<!hw.array<5xi32>>) {
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_element %[[ARRAY1]], %[[INDEX]] : (!llhd.sig<!hw.array<1xi1>>, i32) -> !llhd.sig<i1>
  %0 = llhd.dyn_extract_element %array1, %index : (!llhd.sig<!hw.array<1xi1>>, i32) -> !llhd.sig<i1>
  // CHECK-NEXT: %{{.*}} = llhd.dyn_extract_element %[[ARRAY5]], %[[INDEX]] : (!llhd.sig<!hw.array<5xi32>>, i32) -> !llhd.sig<i32>
  %1 = llhd.dyn_extract_element %array5, %index : (!llhd.sig<!hw.array<5xi32>>, i32) -> !llhd.sig<i32>

  return
}
