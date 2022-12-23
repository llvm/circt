// RUN: circt-opt -lower-std-to-handshake %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @remove_unused_mem(
// CHECK-SAME:                         %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           %[[VAL_0x:.*]] = merge %[[VAL_0]] : none
// CHECK:           return %[[VAL_0x]] : none
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
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_7:.*]] = join %[[VAL_1x]], %[[VAL_2]]#1, %[[VAL_2]]#2 : none, none, none
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_1x]] {value = 11 : i32} : i32
// CHECK:           %[[VAL_3]], %[[VAL_4]] = store {{\[}}%[[VAL_6]]] %[[VAL_8]], %[[VAL_1x]] : index, i32
// CHECK:           %[[VAL_9:.*]] = join %[[VAL_1x]], %[[VAL_2]]#1 : none, none
// CHECK:           %[[VAL_10:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_6]]] %[[VAL_2]]#0, %[[VAL_9]] : index, i32
// CHECK:           return %[[VAL_7]] : none
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
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<10xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<10xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_1x]] {value = 1 : index} : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_1x]] {value = 1 : index} : index
// CHECK:           memref.dma_start %[[VAL_3]]{{\[}}%[[VAL_2]]], %[[VAL_4]]{{\[}}%[[VAL_2]]], %[[VAL_7]], %[[VAL_5]]{{\[}}%[[VAL_6]]] : memref<10xf32>, memref<10xf32>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_5]]{{\[}}%[[VAL_6]]], %[[VAL_7]] : memref<1xi32>
// CHECK:           return %[[VAL_1x]] : none
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
