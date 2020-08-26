// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

// This test verifies that a store operation can be correctly translated.

func @store_only () -> () {
  %c0 = constant 0 : index
  %f0 = constant 1.0 : f32
  %A = alloc() : memref<10xf32>
  affine.store %f0, %A[%c0] : memref<10xf32>
  return
}

// CHECK: handshake.func @store_only(%[[ARG0:.*]]: none, ...) -> none {
// CHECK:   %[[VAL0:.*]] = "handshake.memory"(%[[VAL6:.*]]#0, %[[VAL6]]#1) {id = 0 : i32, ld_count = 0 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index) -> none
// CHECK:   %[[VAL1:.*]]:2 = "handshake.fork"(%[[ARG0]]) {control = true} : (none) -> (none, none)
// CHECK:   %[[VAL2:.*]]:3 = "handshake.fork"(%[[VAL1]]#1) {control = true} : (none) -> (none, none, none)
// CHECK:   %[[VAL3:.*]] = "handshake.join"(%[[VAL2]]#2, %[[VAL0]]) {control = true} : (none, none) -> none
// CHECK:   %[[VAL4:.*]] = "handshake.constant"(%[[VAL2]]#1) {value = 0 : index} : (none) -> index
// CHECK:   %[[VAL5:.*]] = "handshake.constant"(%[[VAL2]]#0) {value = 1.000000e+00 : f32} : (none) -> f32
// CHECK:   %[[VAL6]]:2 = "handshake.store"(%[[VAL5]], %[[VAL4]], %[[VAL1]]#0) : (f32, index, none) -> (f32, index)
// CHECK:   handshake.return %[[VAL3]] : none
// CHECK: }