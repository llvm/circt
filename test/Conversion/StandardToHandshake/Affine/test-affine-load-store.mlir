// RUN: circt-opt %s -create-dataflow -split-input-file | FileCheck %s

// -----

func @load_store () -> () {
  %c0 = constant 0 : index
  %A = alloc() : memref<10xf32>
  %0 = affine.load %A[%c0] : memref<10xf32>
  affine.store %0, %A[%c0] : memref<10xf32>
  return
}

// CHECK: handshake.func @load_store(%[[ARG0:.*]]: none, ...) -> none {
// CHECK:   %[[VAL0:.*]]:3 = "handshake.memory"(%[[VAL9:.*]]#0, %[[VAL9]]#1, %[[ADDR:.*]]) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK:   %[[VAL1:.*]]:2 = "handshake.fork"(%[[VAL0]]#2) {control = false} : (none) -> (none, none)
// CHECK:   %[[VAL2:.*]]:3 = "handshake.fork"(%[[ARG0]]) {control = true} : (none) -> (none, none, none)
// CHECK:   %[[VAL3:.*]]:2 = "handshake.fork"(%[[VAL2]]#2) {control = true} : (none) -> (none, none)
// CHECK:   %[[VAL4:.*]] = "handshake.join"(%[[VAL3]]#1, %[[VAL1]]#1, %[[VAL0]]#1) {control = true} : (none, none, none) -> none
// CHECK:   %[[VAL5:.*]] = "handshake.constant"(%[[VAL3]]#0) {value = 0 : index} : (none) -> index
// CHECK:   %[[VAL6:.*]]:2 = "handshake.fork"(%[[VAL5]]) {control = false} : (index) -> (index, index)
// CHECK:   %[[VAL7:.*]], %[[ADDR]] = "handshake.load"(%[[VAL6]]#0, %[[VAL0]]#0, %[[VAL2]]#1) : (index, f32, none) -> (f32, index)
// CHECK:   %[[VAL8:.*]] = "handshake.join"(%[[VAL2]]#0, %[[VAL1]]#0) {control = true} : (none, none) -> none
// CHECK:   %[[VAL9]]:2 = "handshake.store"(%[[VAL7]], %[[VAL6]]#1, %[[VAL8]]) : (f32, index, none) -> (f32, index)
// CHECK:   handshake.return %[[VAL4]] : none
// CHECK: }
