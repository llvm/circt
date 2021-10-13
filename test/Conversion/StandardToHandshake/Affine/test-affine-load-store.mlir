// RUN: circt-opt %s -lower-std-to-handshake -split-input-file | FileCheck %s

// -----

// Simple load-store pair that has WAR dependence using constant address.

func @load_store () -> () {
  %c0 = constant 0 : index
  %A = memref.alloc() : memref<10xf32>
  %0 = affine.load %A[%c0] : memref<10xf32>
  affine.store %0, %A[%c0] : memref<10xf32>
  return
}

// CHECK: handshake.func @load_store(%[[ARG0:.*]]: none, ...) -> none {
// CHECK:   %[[VAL0:.*]]:3 = "handshake.memory"(%[[VAL9:.*]]#0, %[[VAL9]]#1, %[[ADDR:.*]]) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK:   %[[VAL1:.*]]:2 = "handshake.fork"(%[[VAL0]]#2) {control = true} : (none) -> (none, none)
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

// -----

// Simple load-store pair that has WAR dependence with addresses in affine expressions.

func @affine_map_addr () -> () {
  %c5 = constant 5 : index
  %A = memref.alloc() : memref<10xf32>
  %0 = affine.load %A[%c5 + 1] : memref<10xf32>
  affine.store %0, %A[%c5 - 1] : memref<10xf32>
  return
}

// CHECK:      handshake.func @affine_map_addr(%[[ARG0:.*]]: none, ...) -> none {
// CHECK-NEXT:   %[[VAL0:.*]]:3 = "handshake.memory"(%[[VAL13:.*]]#0, %[[VAL13]]#1, %[[ADDR:.*]]) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK-NEXT:   %[[VAL1:.*]]:2 = "handshake.fork"(%[[VAL0]]#2) {control = true} : (none) -> (none, none)
// CHECK-NEXT:   %[[VAL2:.*]]:3 = "handshake.fork"(%[[ARG0]]) {control = true} : (none) -> (none, none, none)
// CHECK-NEXT:   %[[VAL3:.*]]:4 = "handshake.fork"(%[[VAL2]]#2) {control = true} : (none) -> (none, none, none, none)
// CHECK-NEXT:   %[[VAL4:.*]] = "handshake.join"(%[[VAL3]]#3, %[[VAL1]]#1, %[[VAL0]]#1) {control = true} : (none, none, none) -> none
// CHECK-NEXT:   %[[VAL5:.*]] = "handshake.constant"(%[[VAL3]]#2) {value = 5 : index} : (none) -> index
// CHECK-NEXT:   %[[VAL6:.*]]:2 = "handshake.fork"(%[[VAL5]]) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %[[VAL7:.*]] = "handshake.constant"(%[[VAL3]]#1) {value = 1 : index} : (none) -> index
// CHECK-NEXT:   %[[VAL8:.*]] = addi %[[VAL6]]#0, %[[VAL7]] : index
// CHECK-NEXT:   %[[VAL9:.*]], %[[ADDR]] = "handshake.load"(%[[VAL8]], %[[VAL0]]#0, %[[VAL2]]#1) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:   %[[VAL10:.*]] = "handshake.constant"(%[[VAL3]]#0) {value = -1 : index} : (none) -> index
// CHECK-NEXT:   %[[VAL11:.*]] = addi %[[VAL6]]#1, %[[VAL10]] : index
// CHECK-NEXT:   %[[VAL12:.*]] = "handshake.join"(%[[VAL2]]#0, %[[VAL1]]#0) {control = true} : (none, none) -> none
// CHECK-NEXT:   %[[VAL13]]:2 = "handshake.store"(%[[VAL9]], %[[VAL11]], %[[VAL12]]) : (f32, index, none) -> (f32, index)
// CHECK-NEXT:   handshake.return %[[VAL4]] : none
// CHECK-NEXT: }
