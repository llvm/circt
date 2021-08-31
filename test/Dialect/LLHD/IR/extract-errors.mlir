// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics

func @illegal_array_to_signal(%array : !hw.array<3xi32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'result' have to be both either signless integers, signals or arrays with the same element type}}
  %0 = llhd.extract_slice %array, 0 : !hw.array<3xi32> -> !llhd.sig<!hw.array<3xi32>>

  return
}

// -----

func @illegal_signal_to_array(%sig : !llhd.sig<!hw.array<3xi32>>) {
  // expected-error @+1 {{failed to verify that 'target' and 'result' have to be both either signless integers, signals or arrays with the same element type}}
  %0 = llhd.extract_slice %sig, 0 : !llhd.sig<!hw.array<3xi32>> -> !hw.array<3xi32>

  return
}

// -----

func @illegal_array_element_type_mismatch(%sig : !llhd.sig<!hw.array<3xi32>>) {
  // expected-error @+1 {{failed to verify that 'target' and 'result' have to be both either signless integers, signals or arrays with the same element type}}
  %0 = llhd.extract_slice %sig, 0 : !llhd.sig<!hw.array<3xi32>> -> !llhd.sig<!hw.array<2xi1>>

  return
}

// -----

func @illegal_result_array_too_big(%sig : !llhd.sig<!hw.array<3xi32>>) {
  // expected-error @+1 {{'start' + size of the slice have to be smaller or equal to the 'target' size}}
  %0 = llhd.extract_slice %sig, 0 : !llhd.sig<!hw.array<3xi32>> -> !llhd.sig<!hw.array<4xi32>>

  return
}

// -----

func @illegal_int_to_sig(%c : i32) {
  // expected-error @+1 {{failed to verify that 'target' and 'result' have to be both either signless integers, signals or arrays with the same element type}}
  %0 = llhd.extract_slice %c, 0 : i32 -> !llhd.sig<i10>

  return
}

// -----

func @illegal_sig_to_int(%s : !llhd.sig<i32>) {
  // expected-error @+1 {{failed to verify that 'target' and 'result' have to be both either signless integers, signals or arrays with the same element type}}
  %0 = llhd.extract_slice %s, 0 : !llhd.sig<i32> -> i10

  return
}

// -----

func @illegal_out_too_big(%c : i32) {
  // expected-error @+1 {{failed to verify that 'start' + size of the slice have to be smaller or equal to the 'target' size}}
  %0 = llhd.extract_slice %c, 0 : i32 -> i40

  return
}

// -----

func @illegal_array_to_signal(%array : !hw.array<3xi32>, %index : i32) {
  // expected-error @+1 {{'target' and 'result' types have to match apart from their width}}
  %0 = llhd.dyn_extract_slice %array, %index : (!hw.array<3xi32>, i32) -> !llhd.sig<!hw.array<3xi32>>

  return
}

// -----

func @illegal_signal_to_array(%sig : !llhd.sig<!hw.array<3xi32>>, %index: i32) {
  // expected-error @+1 {{'target' and 'result' types have to match apart from their width}}
  %0 = llhd.dyn_extract_slice %sig, %index : (!llhd.sig<!hw.array<3xi32>>, i32) -> !hw.array<3xi32>

  return
}

// -----

func @illegal_array_element_type_mismatch(%sig : !llhd.sig<!hw.array<3xi32>>, %index : i32) {
  // expected-error @+1 {{'target' and 'result' types have to match apart from their width}}
  %0 = llhd.dyn_extract_slice %sig, %index : (!llhd.sig<!hw.array<3xi32>>, i32) -> !llhd.sig<!hw.array<2xi1>>

  return
}

// -----

func @illegal_result_array_too_big(%sig : !llhd.sig<!hw.array<3xi32>>, %index : i32) {
  // expected-error @+1 {{the result width cannot be larger than the target operand width}}
  %0 = llhd.dyn_extract_slice %sig, %index : (!llhd.sig<!hw.array<3xi32>>, i32) -> !llhd.sig<!hw.array<4xi32>>

  return
}

// -----

func @dyn_extract_slice_illegal_conversion(%s : !llhd.sig<i32>, %i : i1) {
  // expected-error @+1 {{'llhd.dyn_extract_slice' op failed to verify that 'target' and 'result' types have to match apart from their width}}
  %0 = llhd.dyn_extract_slice %s, %i : (!llhd.sig<i32>, i1) -> i10

  return
}

// -----

func @dyn_extract_slice_illegal_out_too_wide(%c : i32, %i : i1) {
  // expected-error @+1 {{'llhd.dyn_extract_slice' op failed to verify that the result width cannot be larger than the target operand width}}
  %0 = llhd.dyn_extract_slice %c, %i : (i32, i1) -> i40

  return
}

// -----

func @dyn_extract_slice_illegal_vec_element_conversion(%c : !hw.array<1xi1>, %i : i1) {
  // expected-error @+1 {{'llhd.dyn_extract_slice' op failed to verify that 'target' and 'result' types have to match apart from their width}}
  %0 = llhd.dyn_extract_slice %c, %i : (!hw.array<1xi1>, i1) -> !hw.array<1xi10>

  return
}

// -----

func @extract_element_array_index_out_of_bounds(%array : !hw.array<3xi1>) {
  // expected-error @+1 {{'index' has to be smaller than the width of the 'target' type}}
  %0 = llhd.extract_element %array, 3 : !hw.array<3xi1> -> i1

  return
}

// -----

func @extract_element_tuple_index_out_of_bounds(%tup : !hw.struct<foo: i1, bar: i2, baz: i3>) {
  // expected-error @+1 {{'index' has to be smaller than the width of the 'target' type}}
  %0 = llhd.extract_element %tup, 3 : !hw.struct<foo: i1, bar: i2, baz: i3> -> i3

  return
}

// -----

func @extract_element_array_type_mismatch(%array : !hw.array<3xi1>) {
  // expected-error @+1 {{'result' type must match the type of 'target' at position 'index', or in case 'target' is a signal, it must be a signal of the underlying type of 'target' at position 'index'}}
  %0 = llhd.extract_element %array, 0 : !hw.array<3xi1> -> i2

  return
}

// -----

func @extract_element_tuple_type_mismatch(%tup : !hw.struct<foo: i1, bar: i2, baz: i3>) {
  // expected-error @+1 {{'result' type must match the type of 'target' at position 'index', or in case 'target' is a signal, it must be a signal of the underlying type of 'target' at position 'index'}}
  %0 = llhd.extract_element %tup, 0 : !hw.struct<foo: i1, bar: i2, baz: i3> -> i2

  return
}

// -----

func @extract_element_signal_type_mismatch(%sig : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // expected-error @+1 {{'result' type must match the type of 'target' at position 'index', or in case 'target' is a signal, it must be a signal of the underlying type of 'target' at position 'index'}}
  %0 = llhd.extract_element %sig, 0 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>> -> !llhd.sig<i2>

  return
}

// -----

func @extract_element_illegal_signal_alias(%sig : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // expected-error @+1 {{'result' type must match the type of 'target' at position 'index', or in case 'target' is a signal, it must be a signal of the underlying type of 'target' at position 'index'}}
  %0 = llhd.extract_element %sig, 0 : !llhd.sig<!hw.struct<foo: i1, bar: i2, baz: i3>> -> i1

  return
}

// -----

func @dyn_extract_element_array_type_mismatch(%index : i2, %array : !hw.array<3xi1>) {
  // expected-error @+1 {{'result' must be the element type of the 'target' array, in case 'target' is a signal of an array, 'result' also is a signal of the array element type}}
  %0 = llhd.dyn_extract_element %array, %index : (!hw.array<3xi1>, i2) -> i2

  return
}

// -----

func @dyn_extract_element_signal_type_mismatch(%index : i2, %sig : !llhd.sig<!hw.array<3xi1>>) {
  // expected-error @+1 {{'result' must be the element type of the 'target' array, in case 'target' is a signal of an array, 'result' also is a signal of the array element type}}
  %0 = llhd.dyn_extract_element %sig, %index : (!llhd.sig<!hw.array<3xi1>>, i2) -> !llhd.sig<i2>

  return
}

// -----

func @dyn_extract_element_illegal_signal_alias(%index : i2, %sig : !llhd.sig<!hw.array<3xi1>>) {
  // expected-error @+1 {{'result' must be the element type of the 'target' array, in case 'target' is a signal of an array, 'result' also is a signal of the array element type}}
  %0 = llhd.dyn_extract_element %sig, %index : (!llhd.sig<!hw.array<3xi1>>, i2) -> i1

  return
}
