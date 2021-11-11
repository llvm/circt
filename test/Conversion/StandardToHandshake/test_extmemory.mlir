// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s

// CHECK-LABEL:   handshake.func @main(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<4xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: none, ...) -> (i32, none) {
// CHECK:           %[[VAL_2:.*]]:2 = handshake.extmemory[ld = 1, st = 0] (%[[VAL_0]] : memref<4xi32>) (%[[VAL_3:.*]]) {id = 0 : i32} : (index) -> (i32, none)
// CHECK:           %[[VAL_4:.*]]:2 = "handshake.fork"(%[[VAL_1]]) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_5:.*]]:2 = "handshake.fork"(%[[VAL_4]]#1) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_6:.*]] = "handshake.join"(%[[VAL_5]]#1, %[[VAL_2]]#1) {control = true} : (none, none) -> none
// CHECK:           %[[VAL_7:.*]] = "handshake.constant"(%[[VAL_5]]#0) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_8:.*]], %[[VAL_3]] = "handshake.load"(%[[VAL_7]], %[[VAL_2]]#0, %[[VAL_4]]#0) : (index, i32, none) -> (i32, index)
// CHECK:           handshake.return %[[VAL_8]], %[[VAL_6]] : i32, none
// CHECK:         }
func @main(%mem : memref<4xi32>) -> i32 {
  %idx = arith.constant 0 : index
  %0 = memref.load %mem[%idx] : memref<4xi32>
  return %0 : i32
}
