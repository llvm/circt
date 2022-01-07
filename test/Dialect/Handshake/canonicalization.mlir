// RUN: circt-opt -split-input-file -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

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

// -----

// CHECK-LABEL:   handshake.func @simple_mux(
// CHECK-SAME:                               %[[VAL_0:.*]]: index,
// CHECK-SAME:                               %[[VAL_1:.*]]: none, ...) -> (none, none)
// CHECK:           return %[[VAL_1]], %[[VAL_1]] : none, none
// CHECK:         }
handshake.func @simple_mux(%c : index, %arg1: none) -> (none, none) {
  %0 = mux %c [%arg1, %arg1, %arg1] : index, none
  handshake.return %0, %arg1 : none, none
}


// -----

// CHECK-LABEL:   handshake.func @simple_mux2(
// CHECK-SAME:                                %[[VAL_0:.*]]: index,
// CHECK-SAME:                                %[[VAL_1:.*]]: none, ...) -> (none, none)
// CHECK:           %[[VAL_2:.*]] = buffer [2] %[[VAL_1]] {sequential = true} : none
// CHECK:           return %[[VAL_2]], %[[VAL_1]] : none, none
// CHECK:         }
handshake.func @simple_mux2(%c : index, %arg1: none) -> (none, none) {
  %0:3 = fork [3] %arg1 : none
  %1 = buffer [2] %0#0 {sequential = true} : none
  %2 = buffer [2] %0#1 {sequential = true} : none
  %3 = mux %c [%1, %0#2] : index, none
  handshake.return %2, %3 : none, none
}

// -----

// CHECK-LABEL:   handshake.func @simple_mux3(
// CHECK-SAME:                                %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                %[[VAL_1:.*]]: index,
// CHECK-SAME:                                %[[VAL_2:.*]]: none, ...) -> (i32, i32, none)
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_0]] : i32
// CHECK:           %[[VAL_4:.*]] = buffer [2] %[[VAL_3]]#1 {sequential = true} : i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_3]]#0, %[[VAL_4]] : i32
// CHECK:           return %[[VAL_5]], %[[VAL_0]], %[[VAL_2]] : i32, i32, none
// CHECK:         }
handshake.func @simple_mux3(%arg0 : i32, %c : index, %arg1: none) -> (i32, i32, none) {
  %0:3 = fork [3] %arg0 : i32
  %1:2 = fork [2] %0#0 : i32
  %3 = mux %c [%1#1, %0#1] : index, i32
  %4 = buffer [2] %1#1 {sequential = true} : i32
  %5 = arith.addi %0#1, %4 : i32
  handshake.return %5, %3, %arg1 : i32, i32, none
}

// -----

// CHECK-LABEL:   handshake.func @fork_to_fork(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                 %[[VAL_1:.*]]: none, ...) -> (i32, i32, i32, none)
// CHECK:           %[[VAL_2:.*]]:3 = fork [3] %[[VAL_0]] : i32
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1, %[[VAL_2]]#2, %[[VAL_1]] : i32, i32, i32, none
// CHECK:         }
handshake.func @fork_to_fork(%arg0 : i32, %arg1: none) -> (i32, i32, i32, none) {
  %0:2 = fork [2] %arg0 : i32
  %1:2 = fork [2] %0#0 : i32
  handshake.return %0#1, %1#0, %1#1, %arg1 : i32, i32, i32, none
}

// -----

// CHECK-LABEL:   handshake.func @sunk_buffer(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                 %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           sink %[[VAL_0]] : i32
// CHECK:           return %[[VAL_1]] : none
// CHECK:         }

handshake.func @sunk_buffer(%arg0 : i32, %arg1: none) -> (none) {
  %0 = buffer [2] %arg0 {sequential  = false} : i32
  sink %0 : i32
  return %arg1 : none
}
