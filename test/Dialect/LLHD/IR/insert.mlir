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

// CHECK-LABEL: @insert_slice_vectors
// CHECK-SAME: %[[VEC2:.*]]: vector<2xi1>
// CHECK-SAME: %[[VEC5:.*]]: vector<5xi1>
func @insert_slice_vectors(%vec2 : vector<2xi1>, %vec5 : vector<5xi1>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[VEC5]], %[[VEC2]], 3 : vector<5xi1>, vector<2xi1>
  %0 = llhd.insert_slice %vec5, %vec2, 3 : vector<5xi1>, vector<2xi1>
  // CHECK-NEXT: %{{.*}} = llhd.insert_slice %[[VEC2]], %[[VEC2]], 0 :  vector<2xi1>, vector<2xi1>
  %1 = llhd.insert_slice %vec2, %vec2, 0 : vector<2xi1>, vector<2xi1>

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

// CHECK-LABEL: @insert_element_vectors
// CHECK-SAME: %[[V1:.*]]: vector<4xi1>,
// CHECK-SAME: %[[V8:.*]]: vector<4xi8>,
// CHECK-SAME: %[[I1:.*]]: i1,
// CHECK-SAME: %[[I8:.*]]: i8
func @insert_element_vectors(%v1 : vector<4xi1>, %v8 : vector<4xi8>, %i1 : i1, %i8 : i8) {
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[V1]], %[[I1]], 0 : vector<4xi1>, i1
  %0 = llhd.insert_element %v1, %i1, 0 : vector<4xi1>, i1
  // CHECK-NEXT: %{{.*}} = llhd.insert_element %[[V8]], %[[I8]], 2 : vector<4xi8>, i8
  %1 = llhd.insert_element %v8, %i8, 2 : vector<4xi8>, i8

  return
}

// -----

func @illegal_kind(%c : i32, %vec : vector<2xi32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or vectors with the same element type}}
  %0 = llhd.insert_slice %vec, %c, 0 : vector<2xi32>, i32

  return
}

// -----

func @illegal_elemental_type(%slice : vector<1xi1>, %vec : vector<2xi32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or vectors with the same element type}}
  %0 = llhd.insert_slice %vec, %slice, 0 : vector<2xi32>, vector<1xi1>

  return
}

// -----

func @insert_slice_illegal_start_index_int(%slice : i16, %c : i32) {
  // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
  %0 = llhd.insert_slice %c, %slice, 20 : i32, i16

  return
}

// -----

func @insert_slice_illegal_start_index_vector(%slice : vector<2xi1>, %vec : vector<3xi1>) {
  // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
  %0 = llhd.insert_slice %vec, %slice, 2 : vector<3xi1>, vector<2xi1>

  return
}

// -----

func @insert_element_index_out_of_bounds(%e : i1, %vec : vector<3xi1>) {
  // expected-error @+1 {{failed to verify that 'index' has to be smaller than the 'target' size}}
  %0 = llhd.insert_element %vec, %e, 3 : vector<3xi1>, i1

  return
}

// -----

func @insert_element_type_mismatch_vector(%e : i2, %vec : vector<3xi1>) {
  // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
  %0 = llhd.insert_element %vec, %e, 0 : vector<3xi1>, i2

  return
}

// -----

func @insert_element_type_mismatch_tuple(%e : i2, %tup : tuple<i2, i1, i2>) {
  // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
  %0 = llhd.insert_element %tup, %e, 1 : tuple<i2, i1, i2>, i2

  return
}
