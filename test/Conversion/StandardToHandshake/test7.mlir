// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
func @simple_loop() {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]] = "handshake.branch"(%[[VAL_0]]) {control = true} : (none) -> none
// CHECK:           %[[VAL_2:.*]]:2 = "handshake.control_merge"(%[[VAL_1]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_3:.*]]:2 = "handshake.fork"(%[[VAL_2]]#0) {control = true} : (none) -> (none, none)
// CHECK:           "handshake.sink"(%[[VAL_2]]#1) : (index) -> ()
// CHECK:           %[[VAL_4:.*]] = "handshake.constant"(%[[VAL_3]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_5:.*]] = "handshake.branch"(%[[VAL_3]]#1) {control = true} : (none) -> none
// CHECK:           %[[VAL_6:.*]] = "handshake.branch"(%[[VAL_4]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_7:.*]]:2 = "handshake.control_merge"(%[[VAL_8:.*]], %[[VAL_5]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_9:.*]]:2 = "handshake.fork"(%[[VAL_7]]#0) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_10:.*]] = "handshake.mux"(%[[VAL_7]]#1, %[[VAL_11:.*]], %[[VAL_6]]) : (index, index, index) -> index
// CHECK:           %[[VAL_12:.*]]:2 = "handshake.fork"(%[[VAL_10]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_13:.*]] = "handshake.constant"(%[[VAL_9]]#0) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_12]]#1, %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]]:2 = "handshake.fork"(%[[VAL_14]]) {control = false} : (i1) -> (i1, i1)
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = "handshake.conditional_branch"(%[[VAL_15]]#1, %[[VAL_9]]#1) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = "handshake.conditional_branch"(%[[VAL_15]]#0, %[[VAL_12]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_19]]) : (index) -> ()
// CHECK:           %[[VAL_20:.*]] = "handshake.merge"(%[[VAL_18]]) : (index) -> index
// CHECK:           %[[VAL_21:.*]]:2 = "handshake.control_merge"(%[[VAL_16]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_22:.*]]:2 = "handshake.fork"(%[[VAL_21]]#0) {control = true} : (none) -> (none, none)
// CHECK:           "handshake.sink"(%[[VAL_21]]#1) : (index) -> ()
// CHECK:           %[[VAL_23:.*]] = "handshake.constant"(%[[VAL_22]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_20]], %[[VAL_23]] : index
// CHECK:           %[[VAL_8]] = "handshake.branch"(%[[VAL_22]]#1) {control = true} : (none) -> none
// CHECK:           %[[VAL_11]] = "handshake.branch"(%[[VAL_24]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_25:.*]]:2 = "handshake.control_merge"(%[[VAL_17]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_25]]#1) : (index) -> ()
// CHECK:           return %[[VAL_25]]#0 : none
// CHECK:         }
// CHECK:       }

^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c1 = arith.constant 1 : index
  br ^bb2(%c1 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %c42 = arith.constant 42 : index
  %1 = arith.cmpi slt, %0, %c42 : index
  cond_br %1, ^bb3, ^bb4
^bb3:	// pred: ^bb2
  %c1_0 = arith.constant 1 : index
  %2 = arith.addi %0, %c1_0 : index
  br ^bb2(%2 : index)
^bb4:	// pred: ^bb2
  return
}
