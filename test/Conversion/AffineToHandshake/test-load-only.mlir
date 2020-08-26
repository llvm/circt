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
// CHECK:   %[[VAL2:.*]]:2 = "handshake.fork"(%[[VAL1]]#1) {control = true} : (none) -> (none, none)
// CHECK:   %[[VAL3:.*]] = "handshake.join"(%[[VAL2]]#1, %[[VAL0]]#1) {control = true} : (none, none) -> none
// CHECK:   %[[VAL4:.*]] = "handshake.constant"(%[[VAL2]]#0) {value = 0 : index} : (none) -> index
// CHECK:   %[[VAL5:.*]], %[[ADDRESS]] = "handshake.load"(%[[VAL4]], %[[VAL0]]#0, %[[VAL1]]#0) : (index, f32, none) -> (f32, index)
// CHECK:   "handshake.sink"(%[[VAL5]]) : (f32) -> ()
// CHECK:   handshake.return %[[VAL3]] : none
// CHECK: }