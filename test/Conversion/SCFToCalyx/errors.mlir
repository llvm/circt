// RUN: circt-opt --lower-scf-to-calyx %s -split-input-file -verify-diagnostics

// expected-error @+1 {{Module contains multiple functions, but no top level function was set. Please see --top-level-function}}
module {
  func.func @f1() {
    return
  }
  func.func @f2() {
    return
  }
}

// -----

func.func @main() {
  cf.br ^bb1
^bb1:
  cf.br ^bb2
^bb2:
  // expected-error @+1 {{CFG backedge detected. Loops must be raised to 'scf.while' or 'scf.for' operations.}}
  cf.br ^bb1
}

// -----
func.func @sum() -> (i32) {
  %sum_0 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index  
  %c10 = arith.constant 10 : index  
  %c1 = arith.constant 1 : index 
  %c1_i32 =  arith.constant 1 : i32 
  %sum = scf.for %iv = %c0 to %c10 step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
    %sum_next = arith.addi %sum_iter, %c1_i32 : i32
    // expected-error @+1 {{Currently do not support non-empty yield operations inside for loops. Run --scf-for-to-while before running --scf-to-calyx.}}
    scf.yield %sum_next : i32
  }
  return %sum : i32
}

// -----
module {
  func.func @main(%upper_bound: index) {
    %alloca_1 = memref.alloca() : memref<40xi32>
    %c0_11 = arith.constant 0 : index
    %c1_13 = arith.constant 1 : index
    %c2_32 = arith.constant 2 : i32
    // expected-error @+1 {{Loop bound not statically known. Should transform into while loop using `--scf-for-to-while` before running --lower-scf-to-calyx.}}
    scf.for %arg0 = %c0_11 to %upper_bound step %c1_13 {
      %0 = memref.load %alloca_1[%arg0] : memref<40xi32>
      %2 = arith.addi %0, %c2_32 : i32
      memref.store %2, %alloca_1[%arg0] : memref<40xi32>
    }
    return
  }
}

// -----

module {
  func.func @main() -> i32 {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cinit = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<6xi32>
    // expected-error @+1 {{Reduce operations in scf.parallel is not supported yet}}
    %r:1 = scf.parallel (%arg2) = (%c0) to (%c3) step (%c1) init (%cinit) -> i32 {
      %6 = memref.load %alloc[%arg2] : memref<6xi32>
      scf.reduce(%6 : i32) {
        ^bb0(%lhs : i32, %rhs: i32):
          %res = arith.addi %lhs, %rhs : i32
          scf.reduce.return %res : i32
      }
    }
    return %r : i32
  }
}

