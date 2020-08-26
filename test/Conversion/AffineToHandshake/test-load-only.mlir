// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

// This test verifies that a load operation can be correctly translated.

func @load_only () -> () {
  %c0 = constant 0 : index
  %A = alloc() : memref<10xf32>
  %0 = affine.load %A[%c0] : memref<10xf32>
  return
}

// CHECK: handshake.func @load_only(%[[ARG0:.*]]: none, ...) -> none {
// CHECK:   %[[VAL0:.*]]:2 = "handshake.memory"(%[[ADDRESS:.*]]) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 0 : i32, type = memref<10xf32>} : (index) -> (f32, none)
// CHECK:   %[[VAL1:.*]]:2 = "handshake.fork"(%[[ARG0]]) {control = true} : (none) -> (none, none)
// CHECK:   %[[VAL2:.*]] = "handshake.join"(%[[VAL1]]#1, %[[VAL0]]#1) {control = true} : (none, none) -> none
// CHECK:   %[[VAL3:.*]] = "handshake.constant"(%[[VAL1]]#0) {value = 0 : index} : (none) -> index
// CHECK:   %[[VAL4:.*]], %[[ADDRESS]] = "handshake.load"(%[[VAL3]], %[[VAL0]]#0, %[[ARG0]]) : (index, f32, none) -> (f32, index)
// CHECK:   "handshake.sink"(%[[VAL4]]) : (f32) -> ()
// CHECK:   handshake.return %[[VAL2]] : none
// CHECK: }