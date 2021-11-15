// RUN: circt-opt -lower-std-to-handshake %s -split-input-file -verify-diagnostics

func @multidim() -> i32 {
  // expected-error @+1 {{memref's must be both statically sized and unidimensional.}}
  %0 = memref.alloc() : memref<2x2xi32>
  %idx = arith.constant 0 : index
  %1 = memref.load %0[%idx, %idx] : memref<2x2xi32>
  return %1 : i32
}

// -----

func @dynsize(%dyn : index) -> i32{
  // expected-error @+1 {{memref's must be both statically sized and unidimensional.}}
  %0 = memref.alloc(%dyn) : memref<?xi32>
  %idx = arith.constant 0 : index
  %1 = memref.load %0[%idx] : memref<?xi32>
  return %1 : i32
}
