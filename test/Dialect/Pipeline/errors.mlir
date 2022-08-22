// RUN: circt-opt %s -split-input-file -verify-diagnostics

func.func @combinational_condition() {
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.alloc() : memref<8xi32>
  // expected-error @+1 {{'pipeline.while' op condition must have a combinational body, found %3 = "memref.load"(%1, %2) : (memref<8xi32>, index) -> i32}}
  pipeline.while II = 1 iter_args(%arg0 = %c0_i32) : (i32) -> () {
    %c0 = arith.constant 0 : index
    %1 = memref.load %0[%c0] : memref<8xi32>
    %2 = arith.cmpi ult, %1, %arg0 : i32
    pipeline.register %2 : i1
  } do {
    pipeline.while.stage start = 0 {
      pipeline.register
    }
    pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func.func @single_condition() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'pipeline.while' op condition must terminate with a single result, found 'i1', 'i1'}}
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0, %arg0 : i1, i1
  } do {
    pipeline.while.stage start = 0 {
      pipeline.register
    }
    pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func.func @boolean_condition() {
  %c0_i32 = arith.constant 0 : i32
  // expected-error @+1 {{'pipeline.while' op condition must terminate with an i1 result, found 'i32'}}
  pipeline.while II = 1 iter_args(%arg0 = %c0_i32) : (i32) -> () {
    pipeline.register %arg0 : i32
  } do {
    pipeline.while.stage start = 0 {
      pipeline.register
    }
    pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func.func @only_stages() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'pipeline.while' op stages must contain at least one stage}}
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0 : i1
  } do {
    pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func.func @only_stages() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'pipeline.while' op stages may only contain 'pipeline.while.stage' or 'pipeline.terminator' ops, found %1 = "arith.addi"(%arg0, %arg0) : (i1, i1) -> i1}}
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0 : i1
  } do {
    %0 = arith.addi %arg0, %arg0 : i1
    pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func.func @mismatched_register_types() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0 : i1
  } do {
    %0 = pipeline.while.stage start = 0 {
      // expected-error @+1 {{'pipeline.register' op operand types ('i1') must match result types ('i2')}}
      pipeline.register %arg0 : i1
    } : i2
    pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func.func @mismatched_iter_args_types() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0 : i1
  } do {
    pipeline.while.stage start = 0 {
      pipeline.register
    }
    // expected-error @+1 {{'pipeline.terminator' op 'iter_args' types () must match pipeline 'iter_args' types ('i1')}}
    pipeline.terminator iter_args(), results() : () -> ()
  }
  return
}

// -----

func.func @invalid_iter_args() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    pipeline.register %arg0 : i1
  } do {
    pipeline.while.stage start = 0 {
      pipeline.register
    }
    // expected-error @+1 {{'pipeline.terminator' op 'iter_args' must be defined by a 'pipeline.while.stage'}}
    pipeline.terminator iter_args(%false), results() : (i1) -> ()
  }
  return
}

// -----

func.func @mismatched_result_types() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    pipeline.register %arg0 : i1
  } do {
    %0 = pipeline.while.stage start = 0 {
      pipeline.register %arg0 : i1
    } : i1
    // expected-error @+1 {{'pipeline.terminator' op 'results' types () must match pipeline result types ('i1')}}
    pipeline.terminator iter_args(%0), results() : (i1) -> ()
  }
  return
}

// -----

func.func @invalid_results() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    pipeline.register %arg0 : i1
  } do {
    %0 = pipeline.while.stage start = 0 {
      pipeline.register %arg0 : i1
    } : i1
    // expected-error @+1 {{'pipeline.terminator' op 'results' must be defined by a 'pipeline.while.stage'}}
    pipeline.terminator iter_args(%0), results(%false) : (i1) -> (i1)
  }
  return
}

// -----

func.func @negative_start() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0 : i1
  } do {
    // expected-error @+1 {{'pipeline.while.stage' op 'start' must be non-negative}}
    %0 = pipeline.while.stage start = -1 {
      pipeline.register %arg0 : i1
    } : i1
    pipeline.terminator iter_args(%0), results() : (i1) -> ()
  }
  return
}

// -----

func.func @non_monotonic_start0() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0 : i1
  } do {
    %0 = pipeline.while.stage start = 0 {
      pipeline.register %arg0 : i1
    } : i1
    // expected-error @+1 {{'pipeline.while.stage' op 'start' must be after previous 'start' (0)}}
    %1 = pipeline.while.stage start = 0 {
      pipeline.register %0 : i1
    } : i1
    pipeline.terminator iter_args(%1), results() : (i1) -> ()
  }
  return
}

// -----

func.func @non_monotonic_start1() {
  %false = arith.constant 0 : i1
  pipeline.while II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    pipeline.register %arg0 : i1
  } do {
    %0 = pipeline.while.stage start = 0 {
      pipeline.register %arg0 : i1
    } : i1
    %1 = pipeline.while.stage start = 1 {
      pipeline.register %0 : i1
    } : i1
    // expected-error @+1 {{'pipeline.while.stage' op 'start' must be after previous 'start' (1)}}
    %2 = pipeline.while.stage start = 0 {
      pipeline.register %1 : i1
    } : i1
    pipeline.terminator iter_args(%2), results() : (i1) -> ()
  }
  return
}
