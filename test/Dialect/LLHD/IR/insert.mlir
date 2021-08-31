// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics | circt-opt |  circt-opt | FileCheck %s

// CHECK-LABEL: @insert_slice_integers
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI32:.*]]: i32
func @insert_slice_integers(%cI1 : i1, %cI32 : i32) {
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[CI1]], %[[CI1]], 0 : i1, i1
  %0 = llhd.insert_slice %cI1, %cI1, 0 : i1, i1
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[CI32]], %[[CI1]], 31 : i32, i1
  %1 = llhd.insert_slice %cI32, %cI1, 31 : i32, i1

  return
}

// CHECK-LABEL: @insert_slice_arrays
// CHECK-SAME: %[[ARRAY2:.*]]: !hw.array<2xi1>
// CHECK-SAME: %[[ARRAY5:.*]]: !hw.array<5xi1>
func @insert_slice_arrays(%array2 : !hw.array<2xi1>, %array5 : !hw.array<5xi1>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[ARRAY5]], %[[ARRAY2]], 3 : !hw.array<5xi1>, !hw.array<2xi1>
  %0 = llhd.insert_slice %array5, %array2, 3 : !hw.array<5xi1>, !hw.array<2xi1>
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[ARRAY2]], %[[ARRAY2]], 0 :  !hw.array<2xi1>, !hw.array<2xi1>
  %1 = llhd.insert_slice %array2, %array2, 0 : !hw.array<2xi1>, !hw.array<2xi1>

  return
}

// CHECK-LABEL: @insert_element_tuples
// CHECK-SAME: %[[TUP:.*]]: !hw.struct<foo: i1, bar: i8>,
// CHECK-SAME: %[[I1:.*]]: i1,
// CHECK-SAME: %[[I8:.*]]: i8
func @insert_element_tuples(%tup : !hw.struct<foo: i1, bar: i8>, %i1 : i1, %i8 : i8) {
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[TUP]], %[[I1]], 0 : !hw.struct<foo: i1, bar: i8>, i1
  %0 = llhd.insert_element %tup, %i1, 0 : !hw.struct<foo: i1, bar: i8>, i1
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[TUP]], %[[I8]], 1 : !hw.struct<foo: i1, bar: i8>, i8
  %1 = llhd.insert_element %tup, %i8, 1 : !hw.struct<foo: i1, bar: i8>, i8

  return
}

// CHECK-LABEL: @insert_element_arrays
// CHECK-SAME: %[[V1:.*]]: !hw.array<4xi1>,
// CHECK-SAME: %[[V8:.*]]: !hw.array<4xi8>,
// CHECK-SAME: %[[I1:.*]]: i1,
// CHECK-SAME: %[[I8:.*]]: i8
func @insert_element_arrays(%v1 : !hw.array<4xi1>, %v8 : !hw.array<4xi8>, %i1 : i1, %i8 : i8) {
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[V1]], %[[I1]], 0 : !hw.array<4xi1>, i1
  %0 = llhd.insert_element %v1, %i1, 0 : !hw.array<4xi1>, i1
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[V8]], %[[I8]], 2 : !hw.array<4xi8>, i8
  %1 = llhd.insert_element %v8, %i8, 2 : !hw.array<4xi8>, i8

  return
}
