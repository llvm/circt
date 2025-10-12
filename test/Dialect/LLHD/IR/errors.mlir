// RUN: circt-opt %s --split-input-file --verify-diagnostics

hw.module @illegal_array_element_type_mismatch(in %sig : !llhd.ref<!hw.array<3xi32>>, in %ind : i2) {
  // expected-error @+1 {{arrays element type must match}}
  %0 = llhd.sig.array_slice %sig at %ind : <!hw.array<3xi32>> -> <!hw.array<2xi1>>
}

// -----

hw.module @illegal_result_array_too_big(in %sig : !llhd.ref<!hw.array<3xi32>>, in %ind : i2) {
  // expected-error @+1 {{width of result type has to be smaller than or equal to the input type}}
  %0 = llhd.sig.array_slice %sig at %ind : <!hw.array<3xi32>> -> <!hw.array<4xi32>>
}

// -----

hw.module @illegal_sig_to_int_to_wide(in %s : !llhd.ref<i32>, in %ind : i5) {
  // expected-error @+1 {{width of result type has to be smaller than or equal to the input type}}
  %0 = llhd.sig.extract %s from %ind : <i32> -> <i64>
}

// -----

hw.module @extract_element_tuple_index_out_of_bounds(in %tup : !llhd.ref<!hw.struct<foo: i1, bar: i2, baz: i3>>) {
  // expected-error @+1 {{invalid field name specified}}
  %0 = llhd.sig.struct_extract %tup["foobar"] : <!hw.struct<foo: i1, bar: i2, baz: i3>>
}

// -----

hw.module @YieldFromFinal(in %arg0: i42) {
  llhd.final {
    // expected-error @below {{'llhd.halt' op has 1 yield operands, but enclosing 'llhd.final' returns 0}}
    llhd.halt %arg0 : i42
  }
}

// -----

hw.module @WaitYieldCount(in %arg0: i42) {
  llhd.process {
    // expected-error @below {{'llhd.wait' op has 1 yield operands, but enclosing 'llhd.process' returns 0}}
    llhd.wait yield (%arg0 : i42), ^bb1
  ^bb1:
    llhd.halt
  }
}

// -----

hw.module @HaltYieldCount(in %arg0: i42) {
  llhd.process {
    // expected-error @below {{'llhd.halt' op has 1 yield operands, but enclosing 'llhd.process' returns 0}}
    llhd.halt %arg0 : i42
  }
}

// -----

hw.module @HaltYieldCount(in %arg0: i42) {
  llhd.combinational {
    // expected-error @below {{'llhd.yield' op has 1 yield operands, but enclosing 'llhd.combinational' returns 0}}
    llhd.yield %arg0 : i42
  }
}

// -----

hw.module @WaitYieldTypes(in %arg0: i42) {
  llhd.process -> i42, i9001 {
    // expected-error @below {{type of yield operand 1 ('i42') does not match enclosing 'llhd.process' result type ('i9001')}}
    llhd.wait yield (%arg0, %arg0 : i42, i42), ^bb1
  ^bb1:
    llhd.halt
  }
}

// -----

hw.module @HaltYieldTypes(in %arg0: i42) {
  llhd.process -> i42, i9001 {
    // expected-error @below {{type of yield operand 1 ('i42') does not match enclosing 'llhd.process' result type ('i9001')}}
    llhd.halt %arg0, %arg0 : i42, i42
  }
}

// -----

hw.module @HaltYieldTypes(in %arg0: i42) {
  llhd.combinational -> i42, i9001 {
    // expected-error @below {{type of yield operand 1 ('i42') does not match enclosing 'llhd.combinational' result type ('i9001')}}
    llhd.yield %arg0, %arg0 : i42, i42
  }
}
