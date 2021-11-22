// RUN: circt-opt -split-input-file --flatten-memref %s | FileCheck %s

// CHECK-LABEL: func @as_func_arg(
func @as_func_arg(%a : memref<4x4xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i] : memref<4x4xi32>
  memref.store %0, %a[%i, %i] : memref<4x4xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL: func @multidim(
func @multidim(%a : memref<4x8x16x32x64xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i, %i, %i, %i] : memref<4x8x16x32x64xi32>
  return %0 : i32
}


// -----

// CHECK-LABEL: func @as_func_ret(
func @as_func_ret(%a : memref<4x4xi32>) -> memref<4x4xi32> {
  return %a : memref<4x4xi32>
}


// -----

// CHECK-LABEL: func @allocs(
func @allocs() -> memref<4x4xi32> {
  %0 = memref.alloc() : memref<4x4xi32>
  return %0 : memref<4x4xi32>
}

// -----

// CHECK-LABEL: func @across_bbs(
func @across_bbs(%i1 : index, %i2 : index, %c : i1) {
  %0 = memref.alloc() : memref<4x4xi32>
  %1 = memref.alloc() : memref<4x4xi32>
  cond_br %c,
    ^bb1(%0, %1 : memref<4x4xi32>, memref<4x4xi32>),
    ^bb1(%1, %0 : memref<4x4xi32>, memref<4x4xi32>)
^bb1(%m1 : memref<4x4xi32>, %m2 : memref<4x4xi32>):
  %2 = memref.load %m1[%i1, %i2] : memref<4x4xi32>
  memref.store %2, %m2[%i1, %i2] : memref<4x4xi32>
  return
}
