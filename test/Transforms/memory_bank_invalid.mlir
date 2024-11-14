// RUN: circt-opt %s -split-input-file -memory-banking="banking-factor=0" -verify-diagnostics

// expected-error@+1 {{banking factor must be greater than 1}}
func.func @bank_one_dim_unroll0(%arg0: memref<8xf32>, %arg1: memref<8xf32>) -> (memref<8xf32>) {
  %mem = memref.alloc() : memref<8xf32>
  affine.parallel (%i) = (0) to (8) {
    %1 = affine.load %arg0[%i] : memref<8xf32>
    %2 = affine.load %arg1[%i] : memref<8xf32>
    %3 = arith.mulf %1, %2 : f32
    affine.store %3, %mem[%i] : memref<8xf32>
  }
  return %mem : memref<8xf32>
}

