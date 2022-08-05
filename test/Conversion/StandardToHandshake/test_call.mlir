// RUN: circt-opt -lower-std-to-handshake -split-input-file %s | FileCheck %s

// CHECK-LABEL:   handshake.func @bar(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i32
// CHECK:           return %[[VAL_2]], %[[VAL_1]] : i32, none
// CHECK:         }
func.func @bar(%0 : i32) -> i32 {
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
func.func @foo(%0 : i32) -> i32 {
  %a1 = call @bar(%0) : (i32) -> i32
  return %a1 : i32
}

// -----

// Branching control flow with calls in each branch.

// CHECK-LABEL:   handshake.func @add(
func.func @add(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL:   handshake.func @sub(
func.func @sub(%arg0 : i32, %arg1: i32) -> i32 {
  %0 = arith.subi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK:   handshake.func @main(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "in1", "in2", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]] : i1
// CHECK:           %[[VAL_7:.*]]:4 = fork [4] %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]]#0 : i1 to index
// CHECK:           %[[VAL_10:.*]] = buffer [2] fifo %[[VAL_8]] : index
// CHECK:           %[[VAL_11:.*]]:2 = fork [2] %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_7]]#3, %[[VAL_4]] : i32
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_7]]#2, %[[VAL_5]] : i32
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = cond_br %[[VAL_7]]#1, %[[VAL_3]] : none
// CHECK:           %[[VAL_18:.*]] = merge %[[VAL_12]] : i32
// CHECK:           %[[VAL_19:.*]] = merge %[[VAL_14]] : i32
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = control_merge %[[VAL_16]] : none
// CHECK:           %[[VAL_22:.*]]:2 = fork [2] %[[VAL_20]] : none
// CHECK:           sink %[[VAL_21]] : index
// CHECK:           %[[VAL_23:.*]]:2 = instance @add(%[[VAL_18]], %[[VAL_19]], %[[VAL_22]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_23]]#1 : none
// CHECK:           %[[VAL_24:.*]] = br %[[VAL_22]]#0 : none
// CHECK:           %[[VAL_25:.*]] = br %[[VAL_23]]#0 : i32
// CHECK:           %[[VAL_26:.*]] = merge %[[VAL_13]] : i32
// CHECK:           %[[VAL_27:.*]] = merge %[[VAL_15]] : i32
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = control_merge %[[VAL_17]] : none
// CHECK:           %[[VAL_30:.*]]:2 = fork [2] %[[VAL_28]] : none
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_31:.*]]:2 = instance @sub(%[[VAL_26]], %[[VAL_27]], %[[VAL_30]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_31]]#1 : none
// CHECK:           %[[VAL_32:.*]] = br %[[VAL_30]]#0 : none
// CHECK:           %[[VAL_33:.*]] = br %[[VAL_31]]#0 : i32
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_11]]#1 {{\[}}%[[VAL_32]], %[[VAL_24]]] : index, none
// CHECK:           %[[VAL_35:.*]] = mux %[[VAL_11]]#0 {{\[}}%[[VAL_33]], %[[VAL_25]]] : index, i32
// CHECK:           return %[[VAL_35]], %[[VAL_34]] : i32, none
// CHECK:         }
func.func @main(%arg0 : i32, %arg1 : i32, %cond : i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = call @add(%arg0, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%0 : i32)
^bb2:
  %1 = call @sub(%arg0, %arg1) : (i32, i32) -> i32
  cf.br ^bb3(%1 : i32)
^bb3(%res : i32):
  return %res : i32
}
