// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
func @simple_loop() {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: none, ...) -> none {
// CHECK:           %[[VAL_1:.*]] = "handshake.branch"(%[[VAL_0]]) {control = true} : (none) -> none
// CHECK:           %[[VAL_2:.*]]:2 = "handshake.control_merge"(%[[VAL_1]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_3:.*]]:2 = "handshake.fork"(%[[VAL_2]]#0) {control = true} : (none) -> (none, none)
// CHECK:           "handshake.sink"(%[[VAL_2]]#1) : (index) -> ()
// CHECK:           %[[VAL_4:.*]] = "handshake.constant"(%[[VAL_3]]#0) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_5:.*]] = "handshake.branch"(%[[VAL_3]]#1) {control = true} : (none) -> none
// CHECK:           %[[VAL_6:.*]] = "handshake.branch"(%[[VAL_4]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_7:.*]] = "handshake.mux"(%[[VAL_8:.*]]#1, %[[VAL_9:.*]], %[[VAL_6]]) : (index, index, index) -> index
// CHECK:           %[[VAL_10:.*]]:3 = "handshake.fork"(%[[VAL_7]]) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_8]]:2 = "handshake.control_merge"(%[[VAL_11:.*]], %[[VAL_5]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_12:.*]] = cmpi slt, %[[VAL_10]]#1, %[[VAL_10]]#2 : index
// CHECK:           %[[VAL_13:.*]]:2 = "handshake.fork"(%[[VAL_12]]) {control = false} : (i1) -> (i1, i1)
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = "handshake.conditional_branch"(%[[VAL_13]]#1, %[[VAL_10]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_15]]) : (index) -> ()
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = "handshake.conditional_branch"(%[[VAL_13]]#0, %[[VAL_8]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_18:.*]] = "handshake.merge"(%[[VAL_14]]) : (index) -> index
// CHECK:           %[[VAL_19:.*]]:2 = "handshake.control_merge"(%[[VAL_16]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_20:.*]]:3 = "handshake.fork"(%[[VAL_19]]#0) {control = true} : (none) -> (none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_19]]#1) : (index) -> ()
// CHECK:           %[[VAL_21:.*]] = "handshake.constant"(%[[VAL_20]]#1) {value = 52 : index} : (none) -> index
// CHECK:           "handshake.sink"(%[[VAL_21]]) : (index) -> ()
// CHECK:           %[[VAL_22:.*]] = "handshake.constant"(%[[VAL_20]]#0) {value = 62 : index} : (none) -> index
// CHECK:           "handshake.sink"(%[[VAL_22]]) : (index) -> ()
// CHECK:           %[[VAL_9]] = "handshake.branch"(%[[VAL_18]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_11]] = "handshake.branch"(%[[VAL_20]]#2) {control = true} : (none) -> none
// CHECK:           %[[VAL_23:.*]]:2 = "handshake.control_merge"(%[[VAL_17]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_23]]#1) : (index) -> ()
// CHECK:           handshake.return %[[VAL_23]]#0 : none
// CHECK:         }
// CHECK:       }

^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c42 = constant 42 : index
  br ^bb2
^bb2:	// 2 preds: ^bb1, ^bb3
  %1 = cmpi slt, %c42, %c42 : index
  cond_br %1, ^bb3, ^bb4
^bb3:	// pred: ^bb2
  %c52 = constant 52 : index
  %c62 = constant 62 : index
  br ^bb2
^bb4:	// pred: ^bb2
  return
}
