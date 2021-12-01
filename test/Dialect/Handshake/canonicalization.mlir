// RUN: circt-opt -split-input-file -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

// CHECK-LABEL:   handshake.func @simple(
// CHECK-SAME:                           %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["arg0"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]] = constant %[[VAL_0]] {value = 1 : index} : index
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_0]] : none
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]]#0 {value = 42 : index} : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : index
// CHECK:           sink %[[VAL_4]] : index
// CHECK:           return %[[VAL_2]]#1 : none
// CHECK:         }
handshake.func @simple(%arg0: none, ...) -> none {
  %0 = constant %arg0 {value = 1 : index} : index
  %1 = br %arg0 : none
  %2 = br %0 : index
  %3 = merge %1 : none
  %4 = merge %2 : index
  %5:2 = fork [2] %3 : none
  %6 = constant %5#0 {value = 42 : index} : index
  %7 = arith.addi %4, %6 : index
  sink %7 : index
  handshake.return %5#1 : none
}

// -----

// CHECK:   handshake.func @cmerge_with_control_used(%[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> (none, index, none) attributes {argNames = ["arg0", "arg1", "arg2"], resNames = ["out0", "out1", "outCtrl"]} {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = control_merge %[[VAL_0]], %[[VAL_1]] : none
// CHECK:           return %[[VAL_3]], %[[VAL_4]], %[[VAL_2]] : none, index, none
// CHECK:         }
handshake.func @cmerge_with_control_used(%arg0: none, %arg1: none, %arg2: none) -> (none, index, none) {
  %result, %index = control_merge %arg0, %arg1 : none
  handshake.return %result, %index, %arg2 : none, index, none
}


// -----

// CHECK:   handshake.func @cmerge_with_control_sunk(%[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> (none, none) attributes {argNames = ["arg0", "arg1", "arg2"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]], %[[VAL_1]] : none
// CHECK:           return %[[VAL_3]], %[[VAL_2]] : none, none
// CHECK:         }
handshake.func @cmerge_with_control_sunk(%arg0: none, %arg1: none, %arg2: none) -> (none, none) {
  %result, %index = control_merge %arg0, %arg1 : none
  sink %index : index
  handshake.return %result, %arg2 : none, none
}

// -----

// CHECK:   handshake.func @cmerge_with_control_ignored(%[[VAL_0:.*]]: none, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> (none, none) attributes {argNames = ["arg0", "arg1", "arg2"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]], %[[VAL_1]] : none
// CHECK:           return %[[VAL_3]], %[[VAL_2]] : none, none
// CHECK:         }
handshake.func @cmerge_with_control_ignored(%arg0: none, %arg1: none, %arg2: none) -> (none, none) {
  %result, %index = control_merge %arg0, %arg1 : none
  handshake.return %result, %arg2 : none, none
}

// -----

// CHECK-LABEL:   handshake.func @sunk_constant(
// CHECK-SAME:                                  %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["arg0"], resNames = ["outCtrl"]} {
// CHECK:           return %[[VAL_0]] : none
// CHECK:         }
handshake.func @sunk_constant(%arg0: none) -> (none) {
  %0 = constant %arg0 { value = 24 : i8 } : i8
  sink %0 : i8
  handshake.return %arg0: none
}

// -----

// CHECK-LABEL:   handshake.func @unused_fork_result(
// CHECK-SAME:                                       %[[VAL_0:.*]]: none, ...) -> (none, none) attributes {argNames = ["arg0"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:2 = fork [2] %[[VAL_0]] : none
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : none, none
// CHECK:         }
handshake.func @unused_fork_result(%arg0: none) -> (none, none) {
  %0:3 = fork [3] %arg0 : none
  handshake.return %0#0, %0#2 : none, none
}
