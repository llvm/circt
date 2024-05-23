// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics

func.func @illegal_signal_to_array(%sig : !hw.inout<array<3xi32>>, %ind: i2) {
  // expected-error @+1 {{'llhd.sig.array_slice' op result #0 must be InOutType of an ArrayType values, but got '!hw.array<3xi32>'}}
  %0 = llhd.sig.array_slice %sig at %ind : (!hw.inout<array<3xi32>>) -> !hw.array<3xi32>

  return
}

// -----

func.func @illegal_array_element_type_mismatch(%sig : !hw.inout<array<3xi32>>, %ind: i2) {
  // expected-error @+1 {{arrays element type must match}}
  %0 = llhd.sig.array_slice %sig at %ind : (!hw.inout<array<3xi32>>) -> !hw.inout<array<2xi1>>

  return
}

// -----

func.func @illegal_result_array_too_big(%sig : !hw.inout<array<3xi32>>, %ind: i2) {
  // expected-error @+1 {{width of result type has to be smaller than or equal to the input type}}
  %0 = llhd.sig.array_slice %sig at %ind : (!hw.inout<array<3xi32>>) -> !hw.inout<array<4xi32>>

  return
}

// -----

func.func @illegal_sig_to_int(%s : !hw.inout<i32>, %ind: i5) {
  // expected-error @+1 {{'llhd.sig.extract' op result #0 must be InOutType of a signless integer bitvector values, but got 'i10'}}
  %0 = llhd.sig.extract %s from %ind : (!hw.inout<i32>) -> i10

  return
}

// -----

func.func @illegal_sig_to_int_to_wide(%s : !hw.inout<i32>, %ind: i5) {
  // expected-error @+1 {{width of result type has to be smaller than or equal to the input type}}
  %0 = llhd.sig.extract %s from %ind : (!hw.inout<i32>) -> !hw.inout<i64>

  return
}

// -----

func.func @extract_element_tuple_index_out_of_bounds(%tup : !hw.inout<struct<foo: i1, bar: i2, baz: i3>>) {
  // expected-error @+1 {{invalid field name specified}}
  %0 = llhd.sig.struct_extract %tup["foobar"] : !hw.inout<struct<foo: i1, bar: i2, baz: i3>>

  return
}
