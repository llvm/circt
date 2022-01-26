// RUN: circt-opt %s -split-input-file -verify-diagnostics

func @combinational_condition() {
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.alloc() : memref<8xi32>
  // expected-error @+1 {{'staticlogic.pipeline.while' op condition must have a combinational body, found %1 = memref.load %0[%c0] : memref<8xi32>}}
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %c0_i32) : (i32) -> () {
    %c0 = arith.constant 0 : index
    %1 = memref.load %0[%c0] : memref<8xi32>
    %2 = arith.cmpi ult, %1, %arg0 : i32
    staticlogic.pipeline.register %2 : i1
  } do {
    staticlogic.pipeline.stage start = 0 {
      staticlogic.pipeline.register
    }
    staticlogic.pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func @single_condition() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'staticlogic.pipeline.while' op condition must terminate with a single result, found 'i1', 'i1'}}
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    staticlogic.pipeline.register %arg0, %arg0 : i1, i1
  } do {
    staticlogic.pipeline.stage start = 0 {
      staticlogic.pipeline.register
    }
    staticlogic.pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func @boolean_condition() {
  %c0_i32 = arith.constant 0 : i32
  // expected-error @+1 {{'staticlogic.pipeline.while' op condition must terminate with an i1 result, found 'i32'}}
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %c0_i32) : (i32) -> () {
    staticlogic.pipeline.register %arg0 : i32
  } do {
    staticlogic.pipeline.stage start = 0 {
      staticlogic.pipeline.register
    }
    staticlogic.pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func @only_stages() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'staticlogic.pipeline.while' op stages must contain at least one stage}}
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    staticlogic.pipeline.register %arg0 : i1
  } do {
    staticlogic.pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func @only_stages() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'staticlogic.pipeline.while' op stages may only contain 'staticlogic.pipeline.stage' or 'staticlogic.pipeline.terminator' ops, found %0 = arith.addi %arg0, %arg0 : i1}}
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    staticlogic.pipeline.register %arg0 : i1
  } do {
    %0 = arith.addi %arg0, %arg0 : i1
    staticlogic.pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func @mismatched_register_types() {
  %false = arith.constant 0 : i1
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    staticlogic.pipeline.register %arg0 : i1
  } do {
    %0 = staticlogic.pipeline.stage start = 0 {
      // expected-error @+1 {{'staticlogic.pipeline.register' op operand types ('i1') must match result types ('i2')}}
      staticlogic.pipeline.register %arg0 : i1
    } : i2
    staticlogic.pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func @mismatched_iter_args_types() {
  %false = arith.constant 0 : i1
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    staticlogic.pipeline.register %arg0 : i1
  } do {
    staticlogic.pipeline.stage start = 0 {
      staticlogic.pipeline.register
    }
    // expected-error @+1 {{'staticlogic.pipeline.terminator' op 'iter_args' types () must match pipeline 'iter_args' types ('i1')}}
    staticlogic.pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func @invalid_iter_args() {
  %false = arith.constant 0 : i1
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    staticlogic.pipeline.register %arg0 : i1
  } do {
    staticlogic.pipeline.stage start = 0 {
      staticlogic.pipeline.register
    }
    // expected-error @+1 {{'staticlogic.pipeline.terminator' op 'iter_args' must be defined by a 'staticlogic.pipeline.stage'}}
    staticlogic.pipeline.terminator iter_args(%false), results() : (i1) -> ()
  }
  return
}

// -----

func @mismatched_result_types() {
  %false = arith.constant 0 : i1
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    staticlogic.pipeline.register %arg0 : i1
  } do {
    %0 = staticlogic.pipeline.stage start = 0 {
      staticlogic.pipeline.register %arg0 : i1
    } : i1
    // expected-error @+1 {{'staticlogic.pipeline.terminator' op 'results' types () must match pipeline result types ('i1')}}
    staticlogic.pipeline.terminator iter_args(%0), results() : (i1) -> ()
  }
  return
}

// -----

func @invalid_results() {
  %false = arith.constant 0 : i1
  staticlogic.pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    staticlogic.pipeline.register %arg0 : i1
  } do {
    %0 = staticlogic.pipeline.stage start = 0 {
      staticlogic.pipeline.register %arg0 : i1
    } : i1
    // expected-error @+1 {{'staticlogic.pipeline.terminator' op 'results' must be defined by a 'staticlogic.pipeline.stage'}}
    staticlogic.pipeline.terminator iter_args(%0), results(%false) : (i1) -> (i1)
  }
  return
}
