// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics

func @illegal_kind(%c : i32, %array : !hw.array<2xi32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or arrays with the same element type}}
  %0 = llhd.insert_slice %array, %c, 0 : !hw.array<2xi32>, i32

  return
}

// -----

func @illegal_elemental_type(%slice : !hw.array<1xi1>, %array : !hw.array<2xi32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or arrays with the same element type}}
  %0 = llhd.insert_slice %array, %slice, 0 : !hw.array<2xi32>, !hw.array<1xi1>

  return
}

// -----

func @insert_slice_illegal_start_index_int(%slice : i16, %c : i32) {
  // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
  %0 = llhd.insert_slice %c, %slice, 20 : i32, i16

  return
}

// -----

func @insert_slice_illegal_start_index_array(%slice : !hw.array<2xi1>, %array : !hw.array<3xi1>) {
  // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
  %0 = llhd.insert_slice %array, %slice, 2 : !hw.array<3xi1>, !hw.array<2xi1>

  return
}

// -----

func @insert_element_index_out_of_bounds(%e : i1, %array : !hw.array<3xi1>) {
  // expected-error @+1 {{failed to verify that 'index' has to be smaller than the 'target' size}}
  %0 = llhd.insert_element %array, %e, 3 : !hw.array<3xi1>, i1

  return
}

// -----

func @insert_element_type_mismatch_array(%e : i2, %array : !hw.array<3xi1>) {
  // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
  %0 = llhd.insert_element %array, %e, 0 : !hw.array<3xi1>, i2

  return
}

// -----

func @insert_element_type_mismatch_tuple(%e : i2, %tup : !hw.struct<foo: i2, bar: i1, baz: i2>) {
  // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
  %0 = llhd.insert_element %tup, %e, 1 : !hw.struct<foo: i2, bar: i1, baz: i2>, i2

  return
}
