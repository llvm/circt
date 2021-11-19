// RUN: circt-opt %s -lower-std-to-handshake -split-input-file | FileCheck %s

// -----

// Simple load-store pair that has WAR dependence using constant address.

// CHECK-LABEL:   handshake.func @load_store(
// CHECK-SAME:                               %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:3 = memory[ld = 1, st = 1] (%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index, index) -> (f32, none, none)
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_1]]#2 : none
// CHECK:           %[[VAL_6:.*]]:3 = fork [3] %[[VAL_0]] : none
// CHECK:           %[[VAL_7:.*]]:2 = fork [2] %[[VAL_6]]#2 : none
// CHECK:           %[[VAL_8:.*]] = join %[[VAL_7]]#1, %[[VAL_5]]#1, %[[VAL_1]]#1 : none
// CHECK:           %[[VAL_9:.*]] = constant %[[VAL_7]]#0 {value = 0 : index} : index
// CHECK:           %[[VAL_10:.*]]:2 = fork [2] %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]], %[[VAL_4]] = load {{\[}}%[[VAL_10]]#0] %[[VAL_1]]#0, %[[VAL_6]]#1 : index, f32
// CHECK:           %[[VAL_12:.*]] = join %[[VAL_6]]#0, %[[VAL_5]]#0 : none
// CHECK:           %[[VAL_2]], %[[VAL_3]] = store {{\[}}%[[VAL_10]]#1] %[[VAL_11]], %[[VAL_12]] : index, f32
// CHECK:           return %[[VAL_8]] : none
// CHECK:         }
func @load_store () -> () {
  %c0 = arith.constant 0 : index
  %A = memref.alloc() : memref<10xf32>
  %0 = affine.load %A[%c0] : memref<10xf32>
  affine.store %0, %A[%c0] : memref<10xf32>
  return
}


// -----

// Simple load-store pair that has WAR dependence with addresses in affine expressions.

// CHECK-LABEL:   handshake.func @affine_map_addr(
// CHECK-SAME:                                    %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:3 = memory[ld = 1, st = 1] (%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index, index) -> (f32, none, none)
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_1]]#2 : none
// CHECK:           %[[VAL_6:.*]]:3 = fork [3] %[[VAL_0]] : none
// CHECK:           %[[VAL_7:.*]]:4 = fork [4] %[[VAL_6]]#2 : none
// CHECK:           %[[VAL_8:.*]] = join %[[VAL_7]]#3, %[[VAL_5]]#1, %[[VAL_1]]#1 : none
// CHECK:           %[[VAL_9:.*]] = constant %[[VAL_7]]#2 {value = 5 : index} : index
// CHECK:           %[[VAL_10:.*]]:2 = fork [2] %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_7]]#1 {value = 1 : index} : index
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_10]]#0, %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]], %[[VAL_4]] = load {{\[}}%[[VAL_12]]] %[[VAL_1]]#0, %[[VAL_6]]#1 : index, f32
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_7]]#0 {value = -1 : index} : index
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_10]]#1, %[[VAL_14]] : index
// CHECK:           %[[VAL_16:.*]] = join %[[VAL_6]]#0, %[[VAL_5]]#0 : none
// CHECK:           %[[VAL_2]], %[[VAL_3]] = store {{\[}}%[[VAL_15]]] %[[VAL_13]], %[[VAL_16]] : index, f32
// CHECK:           return %[[VAL_8]] : none
// CHECK:         }
func @affine_map_addr () -> () {
  %c5 = arith.constant 5 : index
  %A = memref.alloc() : memref<10xf32>
  %0 = affine.load %A[%c5 + 1] : memref<10xf32>
  affine.store %0, %A[%c5 - 1] : memref<10xf32>
  return
}

