// RUN: circt-opt %s --split-input-file --verify-diagnostics

hw.module @errors(in %in0: i32, out out0: i8) {
  // expected-error @below {{requires the same type for all operands and results}}
  %0 = "llhd.delay"(%in0) {delay = #llhd.time<0ns, 1d, 0e>} : (i32) -> i8
  hw.output %0 : i8
}

// -----

hw.module @illegal_signal_to_array(inout %sig : !hw.array<3xi32>, in %ind : i2) {
  // expected-error @+1 {{'llhd.sig.array_slice' op result #0 must be InOutType of an ArrayType values, but got '!hw.array<3xi32>'}}
  %0 = llhd.sig.array_slice %sig at %ind : (!hw.inout<array<3xi32>>) -> !hw.array<3xi32>
}

// -----

hw.module @illegal_array_element_type_mismatch(inout %sig : !hw.array<3xi32>, in %ind : i2) {
  // expected-error @+1 {{arrays element type must match}}
  %0 = llhd.sig.array_slice %sig at %ind : (!hw.inout<array<3xi32>>) -> !hw.inout<array<2xi1>>
}

// -----

hw.module @illegal_result_array_too_big(inout %sig : !hw.array<3xi32>, in %ind : i2) {
  // expected-error @+1 {{width of result type has to be smaller than or equal to the input type}}
  %0 = llhd.sig.array_slice %sig at %ind : (!hw.inout<array<3xi32>>) -> !hw.inout<array<4xi32>>
}

// -----

hw.module @illegal_sig_to_int(inout %s : i32, in %ind : i5) {
  // expected-error @+1 {{'llhd.sig.extract' op result #0 must be InOutType of a signless integer bitvector values, but got 'i10'}}
  %0 = llhd.sig.extract %s from %ind : (!hw.inout<i32>) -> i10
}

// -----

hw.module @illegal_sig_to_int_to_wide(inout %s : i32, in %ind : i5) {
  // expected-error @+1 {{width of result type has to be smaller than or equal to the input type}}
  %0 = llhd.sig.extract %s from %ind : (!hw.inout<i32>) -> !hw.inout<i64>
}

// -----

hw.module @extract_element_tuple_index_out_of_bounds(inout %tup : !hw.struct<foo: i1, bar: i2, baz: i3>) {
  // expected-error @+1 {{invalid field name specified}}
  %0 = llhd.sig.struct_extract %tup["foobar"] : !hw.inout<struct<foo: i1, bar: i2, baz: i3>>
}

// -----

// expected-note @+1 {{prior use here}}
func.func @check_illegal_store(%i1Ptr : !llhd.ptr<i1>, %i32Const : i32) {
  // expected-error @+1 {{use of value '%i32Const' expects different type than prior uses: 'i1' vs 'i32'}}
  llhd.store %i1Ptr, %i32Const : !llhd.ptr<i1>

  return
}

// -----

// expected-error @+1 {{unknown  type `illegaltype` in dialect `llhd`}}
func.func @illegaltype(%arg0: !llhd.illegaltype) {
    return
}

// -----

// expected-error @+2 {{unknown attribute `illegalattr` in dialect `llhd`}}
func.func @illegalattr() {
    %0 = llhd.constant_time #llhd.illegalattr : i1
    return
}

// -----

// expected-error @+3 {{failed to verify that type of 'init' and underlying type of 'signal' have to match.}}
hw.module @check_illegal_sig() {
  %cI1 = hw.constant 0 : i1
  %sig1 = "llhd.sig"(%cI1) {name="foo"} : (i1) -> !hw.inout<i32>
}

// -----

// expected-error @+2 {{failed to verify that type of 'result' and underlying type of 'signal' have to match.}}
hw.module @check_illegal_prb(inout %sig : i1) {
  %prb = "llhd.prb"(%sig) {} : (!hw.inout<i1>) -> i32
}

// -----

// expected-error @+4 {{failed to verify that type of 'value' and underlying type of 'signal' have to match.}}
hw.module @check_illegal_drv(inout %sig : i1) {
  %c = hw.constant 0 : i32
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  "llhd.drv"(%sig, %c, %time) {} : (!hw.inout<i1>, i32, !llhd.time) -> ()
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
