// RUN: circt-opt -lower-std-to-handshake %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @remove_unused_mem(
// CHECK-SAME:                         %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           return %[[VAL_0]] : none
// CHECK:         }
func.func @remove_unused_mem() {
  %0 = memref.alloc() : memref<100xf32>
  return
}

// -----

// CHECK-LABEL:   handshake.func @load_store(
// CHECK-SAME:                               %[[VAL_0:.*]]: index,
// CHECK-SAME:                               %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:3 = memory[ld = 1, st = 1] (%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) {id = 0 : i32, lsq = false} : memref<4xi32>, (i32, index, index) -> (i32, none, none)
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_2]]#1 : none
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_8:.*]]:2 = fork [2] %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]]:3 = fork [3] %[[VAL_1]] : none
// CHECK:           %[[VAL_10:.*]]:2 = fork [2] %[[VAL_9]]#2 : none
// CHECK:           %[[VAL_11:.*]] = join %[[VAL_10]]#1, %[[VAL_6]]#1, %[[VAL_2]]#2 : none, none, none
// CHECK:           %[[VAL_12:.*]] = constant %[[VAL_10]]#0 {value = 11 : i32} : i32
// CHECK:           %[[VAL_3]], %[[VAL_4]] = store {{\[}}%[[VAL_8]]#1] %[[VAL_12]], %[[VAL_9]]#1 : index, i32
// CHECK:           %[[VAL_13:.*]] = join %[[VAL_9]]#0, %[[VAL_6]]#0 : none, none
// CHECK:           %[[VAL_14:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_8]]#0] %[[VAL_2]]#0, %[[VAL_13]] : index, i32
// CHECK:           sink %[[VAL_14]] : i32
// CHECK:           return %[[VAL_11]] : none
// CHECK:         }
func.func @load_store(%1 : index) {
  %0 = memref.alloc() : memref<4xi32>
  %c1 = arith.constant 11 : i32
  memref.store %c1, %0[%1] : memref<4xi32>
  %3 = memref.load %0[%1] : memref<4xi32>
  return
}

// -----

// CHECK-LABEL:   handshake.func @dma(
// CHECK-SAME:                             %[[VAL_0:.*]]: index,
// CHECK-SAME:                             %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]]:3 = fork [3] %[[VAL_1]] : none
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<10xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<10xf32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_8:.*]]:2 = fork [2] %[[VAL_7]] : memref<1xi32>
// CHECK:           %[[VAL_9:.*]] = constant %[[VAL_4]]#1 {value = 1 : index} : index
// CHECK:           %[[VAL_10:.*]]:2 = fork [2] %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_4]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : index
// CHECK:           memref.dma_start %[[VAL_5]]{{\[}}%[[VAL_3]]#0], %[[VAL_6]]{{\[}}%[[VAL_3]]#1], %[[VAL_12]]#0, %[[VAL_8]]#1{{\[}}%[[VAL_10]]#0] : memref<10xf32>, memref<10xf32>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_8]]#0{{\[}}%[[VAL_10]]#1], %[[VAL_12]]#1 : memref<1xi32>
// CHECK:           return %[[VAL_4]]#2 : none
// CHECK:         }
func.func @dma(%1 : index) {
  %mem0 = memref.alloc() : memref<10xf32>
  %mem1 = memref.alloc() : memref<10xf32>
  %tag = memref.alloc() : memref<1xi32>
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  memref.dma_start %mem0[%1], %mem1[%1], %c1, %tag[%c0] : memref<10xf32>, memref<10xf32>, memref<1xi32>
  memref.dma_wait %tag[%c0], %c1 : memref<1xi32>
  return
}
