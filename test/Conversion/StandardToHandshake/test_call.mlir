// RUN: circt-opt -lower-std-to-handshake -split-input-file %s | FileCheck %s

// CHECK-LABEL:   handshake.func @bar(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i32
// CHECK:           return %[[VAL_2]], %[[VAL_1]] : i32, none
// CHECK:         }
func @bar(%0 : i32) -> i32 {
  return %0 : i32
}

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]]:2 = instance @bar(%[[VAL_2]], %[[VAL_3]]#0) : (i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_4]]#1 : none
// CHECK:           return %[[VAL_4]]#0, %[[VAL_3]]#1 : i32, none
// CHECK:         }
func @foo(%0 : i32) -> i32 {
  %a1 = call @bar(%0) : (i32) -> i32
  return %a1 : i32
}

// -----

// Branching control flow with calls in each branch.

// CHECK-LABEL:   handshake.func @add(
func @add(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL:   handshake.func @sub(
func @sub(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.subi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK:   handshake.func @main(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "in1", "in2", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]] : i1
// CHECK:           %[[VAL_7:.*]]:3 = fork [3] %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_7]]#2, %[[VAL_4]] : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_7]]#1, %[[VAL_5]] : i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_7]]#0, %[[VAL_3]] : none
// CHECK:           %[[VAL_14:.*]] = merge %[[VAL_8]] : i32
// CHECK:           %[[VAL_15:.*]] = merge %[[VAL_10]] : i32
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = control_merge %[[VAL_12]] : none
// CHECK:           %[[VAL_18:.*]]:2 = fork [2] %[[VAL_16]] : none
// CHECK:           sink %[[VAL_17]] : index
// CHECK:           %[[VAL_19:.*]]:2 = instance @add(%[[VAL_14]], %[[VAL_15]], %[[VAL_18]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_19]]#1 : none
// CHECK:           %[[VAL_20:.*]] = br %[[VAL_18]]#0 : none
// CHECK:           %[[VAL_21:.*]] = br %[[VAL_19]]#0 : i32
// CHECK:           %[[VAL_22:.*]] = merge %[[VAL_9]] : i32
// CHECK:           %[[VAL_23:.*]] = merge %[[VAL_11]] : i32
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = control_merge %[[VAL_13]] : none
// CHECK:           %[[VAL_26:.*]]:2 = fork [2] %[[VAL_24]] : none
// CHECK:           sink %[[VAL_25]] : index
// CHECK:           %[[VAL_27:.*]]:2 = instance @sub(%[[VAL_22]], %[[VAL_23]], %[[VAL_26]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_27]]#1 : none
// CHECK:           %[[VAL_28:.*]] = br %[[VAL_26]]#0 : none
// CHECK:           %[[VAL_29:.*]] = br %[[VAL_27]]#0 : i32
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = control_merge %[[VAL_28]], %[[VAL_20]] : none
// CHECK:           %[[VAL_32:.*]] = mux %[[VAL_31]] {{\[}}%[[VAL_29]], %[[VAL_21]]] : index, i32
// CHECK:           return %[[VAL_32]], %[[VAL_30]] : i32, none
// CHECK:         }
func @main(%arg0 : i32, %arg1 : i32, %cond : i1) -> i32 {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = call @add(%arg0, %arg1) : (i32, i32) -> i32
  br ^bb3(%0 : i32)
^bb2:
  %1 = call @sub(%arg0, %arg1) : (i32, i32) -> i32
  br ^bb3(%1 : i32)
^bb3(%res : i32):
  return %res : i32
}
