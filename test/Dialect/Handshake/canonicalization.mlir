// RUN: circt-opt -split-input-file -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL:   handshake.func @simple(
// CHECK-SAME:        %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           %[[VAL_1:.*]] = "handshake.constant"(%[[VAL_0]]) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_2:.*]]:2 = "handshake.fork"(%[[VAL_0]]) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_3:.*]] = "handshake.constant"(%[[VAL_2]]#0) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           "handshake.sink"(%[[VAL_4]]) : (index) -> ()
// CHECK:           handshake.return %[[VAL_2]]#1 : none
// CHECK:         }
handshake.func @simple(%arg0: none, ...) -> none {
  %0 = "handshake.constant"(%arg0) {value = 1 : index} : (none) -> index
  %1 = "handshake.branch"(%arg0) {control = true} : (none) -> none
  %2 = "handshake.branch"(%0) {control = false} : (index) -> index
  %3 = "handshake.merge"(%1) : (none) -> none
  %4 = "handshake.merge"(%2) : (index) -> index
  %5:2 = "handshake.fork"(%3) {control = true} : (none) -> (none, none)
  %6 = "handshake.constant"(%5#0) {value = 42 : index} : (none) -> index
  %7 = arith.addi %4, %6 : index
  "handshake.sink"(%7) : (index) -> ()
  handshake.return %5#1 : none
}

// -----

// CHECK-LABEL:   handshake.func @cmerge_with_control_used(
// CHECK-SAME:        %[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> (none, index, none)
// CHECK:           %[[VAL_3:.*]]:2 = "handshake.control_merge"(%[[VAL_0]], %[[VAL_1]]) {control = true} : (none, none) -> (none, index)
// CHECK:           handshake.return %[[VAL_3]]#0, %[[VAL_3]]#1, %[[VAL_2]] : none, index, none
// CHECK:         }
handshake.func @cmerge_with_control_used(%arg0: none, %arg1: none, %arg2: none) -> (none, index, none) {
  %result, %index = "handshake.control_merge"(%arg0, %arg1) {control = true} : (none, none) -> (none, index)
  handshake.return %result, %index, %arg2 : none, index, none
}


// -----

// CHECK-LABEL:   handshake.func @cmerge_with_control_sunk(
// CHECK-SAME:        %[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> (none, none)
// CHECK:           %[[VAL_3:.*]] = "handshake.merge"(%[[VAL_0]], %[[VAL_1]]) : (none, none) -> none
// CHECK:           handshake.return %[[VAL_3]], %[[VAL_2]] : none, none
// CHECK:         }
handshake.func @cmerge_with_control_sunk(%arg0: none, %arg1: none, %arg2: none) -> (none, none) {
  %result, %index = "handshake.control_merge"(%arg0, %arg1) {control = true} : (none, none) -> (none, index)
  "handshake.sink"(%index) : (index) -> ()
  handshake.return %result, %arg2 : none, none
}

// -----

// CHECK-LABEL:   handshake.func @cmerge_with_control_ignored(
// CHECK-SAME:        %[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> (none, none)
// CHECK:           %[[VAL_3:.*]] = "handshake.merge"(%[[VAL_0]], %[[VAL_1]]) : (none, none) -> none
// CHECK:           handshake.return %[[VAL_3]], %[[VAL_2]] : none, none
// CHECK:         }
handshake.func @cmerge_with_control_ignored(%arg0: none, %arg1: none, %arg2: none) -> (none, none) {
  %result, %index = "handshake.control_merge"(%arg0, %arg1) {control = true} : (none, none) -> (none, index)
  handshake.return %result, %arg2 : none, none
}

// -----

// CHECK-LABEL:   handshake.func @sunk_constant(
// CHECK-SAME:        %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           handshake.return %[[VAL_0]] : none
// CHECK:         }
handshake.func @sunk_constant(%arg0: none) -> (none) {
  %0 = "handshake.constant"(%arg0) { value = 24 : i8 } : (none) -> i8
  "handshake.sink"(%0) : (i8) -> ()
  handshake.return %arg0: none
}
