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
// CHECK-SAME: %[[ARRAY2:.*]]: !llhd.array<2xi1>
// CHECK-SAME: %[[ARRAY5:.*]]: !llhd.array<5xi1>
func @insert_slice_arrays(%array2 : !llhd.array<2xi1>, %array5 : !llhd.array<5xi1>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[ARRAY5]], %[[ARRAY2]], 3 : !llhd.array<5xi1>, !llhd.array<2xi1>
  %0 = llhd.insert_slice %array5, %array2, 3 : !llhd.array<5xi1>, !llhd.array<2xi1>
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[ARRAY2]], %[[ARRAY2]], 0 :  !llhd.array<2xi1>, !llhd.array<2xi1>
  %1 = llhd.insert_slice %array2, %array2, 0 : !llhd.array<2xi1>, !llhd.array<2xi1>

  return
}

// CHECK-LABEL: @insert_element_tuples
// CHECK-SAME: %[[TUP:.*]]: tuple<i1, i8>,
// CHECK-SAME: %[[I1:.*]]: i1,
// CHECK-SAME: %[[I8:.*]]: i8
func @insert_element_tuples(%tup : tuple<i1, i8>, %i1 : i1, %i8 : i8) {
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[TUP]], %[[I1]], 0 : tuple<i1, i8>, i1
  %0 = llhd.insert_element %tup, %i1, 0 : tuple<i1, i8>, i1
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[TUP]], %[[I8]], 1 : tuple<i1, i8>, i8
  %1 = llhd.insert_element %tup, %i8, 1 : tuple<i1, i8>, i8

  return
}

// CHECK-LABEL: @insert_element_arrays
// CHECK-SAME: %[[V1:.*]]: !llhd.array<4xi1>,
// CHECK-SAME: %[[V8:.*]]: !llhd.array<4xi8>,
// CHECK-SAME: %[[I1:.*]]: i1,
// CHECK-SAME: %[[I8:.*]]: i8
func @insert_element_arrays(%v1 : !llhd.array<4xi1>, %v8 : !llhd.array<4xi8>, %i1 : i1, %i8 : i8) {
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[V1]], %[[I1]], 0 : !llhd.array<4xi1>, i1
  %0 = llhd.insert_element %v1, %i1, 0 : !llhd.array<4xi1>, i1
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[V8]], %[[I8]], 2 : !llhd.array<4xi8>, i8
  %1 = llhd.insert_element %v8, %i8, 2 : !llhd.array<4xi8>, i8

  return
}

// -----

func @illegal_kind(%c : i32, %array : !llhd.array<2xi32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or arrays with the same element type}}
  %0 = llhd.insert_slice %array, %c, 0 : !llhd.array<2xi32>, i32

  return
}

// -----

func @illegal_elemental_type(%slice : !llhd.array<1xi1>, %array : !llhd.array<2xi32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or arrays with the same element type}}
  %0 = llhd.insert_slice %array, %slice, 0 : !llhd.array<2xi32>, !llhd.array<1xi1>

  return
}

// -----

func @insert_slice_illegal_start_index_int(%slice : i16, %c : i32) {
  // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
  %0 = llhd.insert_slice %c, %slice, 20 : i32, i16

  return
}

// -----

func @insert_slice_illegal_start_index_array(%slice : !llhd.array<2xi1>, %array : !llhd.array<3xi1>) {
  // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
  %0 = llhd.insert_slice %array, %slice, 2 : !llhd.array<3xi1>, !llhd.array<2xi1>

  return
}

// -----

func @insert_element_index_out_of_bounds(%e : i1, %array : !llhd.array<3xi1>) {
  // expected-error @+1 {{failed to verify that 'index' has to be smaller than the 'target' size}}
  %0 = llhd.insert_element %array, %e, 3 : !llhd.array<3xi1>, i1

  return
}

// -----

func @insert_element_type_mismatch_array(%e : i2, %array : !llhd.array<3xi1>) {
  // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
  %0 = llhd.insert_element %array, %e, 0 : !llhd.array<3xi1>, i2

  return
}

// -----

func @insert_element_type_mismatch_tuple(%e : i2, %tup : tuple<i2, i1, i2>) {
  // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
  %0 = llhd.insert_element %tup, %e, 1 : tuple<i2, i1, i2>, i2

  return
}
