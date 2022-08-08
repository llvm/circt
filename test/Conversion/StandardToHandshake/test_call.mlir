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

// CHECK:   handshake.func @main(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i1, %[[VAL_3:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]] : i1
// CHECK:           %[[VAL_7:.*]]:4 = fork [4] %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_9:.*]]#0 {value = true} : i1
// CHECK:           %[[VAL_10:.*]] = arith.xori %[[VAL_7]]#0, %[[VAL_8]] : i1
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : i1 to index
// CHECK:           %[[VAL_12:.*]] = buffer [2] fifo %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]]:2 = fork [2] %[[VAL_12]] : index
// CHECK:           %[[VAL_9]]:2 = fork [2] %[[VAL_3]] : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_7]]#3, %[[VAL_4]] : i32
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = cond_br %[[VAL_7]]#2, %[[VAL_5]] : i32
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = cond_br %[[VAL_7]]#1, %[[VAL_9]]#1 : none
// CHECK:           %[[VAL_20:.*]] = merge %[[VAL_14]] : i32
// CHECK:           %[[VAL_21:.*]] = merge %[[VAL_16]] : i32
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = control_merge %[[VAL_18]] : none
// CHECK:           %[[VAL_24:.*]]:2 = fork [2] %[[VAL_22]] : none
// CHECK:           sink %[[VAL_23]] : index
// CHECK:           %[[VAL_25:.*]]:2 = instance @add(%[[VAL_20]], %[[VAL_21]], %[[VAL_24]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_25]]#1 : none
// CHECK:           %[[VAL_26:.*]] = br %[[VAL_24]]#0 : none
// CHECK:           %[[VAL_27:.*]] = br %[[VAL_25]]#0 : i32
// CHECK:           %[[VAL_28:.*]] = merge %[[VAL_15]] : i32
// CHECK:           %[[VAL_29:.*]] = merge %[[VAL_17]] : i32
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = control_merge %[[VAL_19]] : none
// CHECK:           %[[VAL_32:.*]]:2 = fork [2] %[[VAL_30]] : none
// CHECK:           sink %[[VAL_31]] : index
// CHECK:           %[[VAL_33:.*]]:2 = instance @sub(%[[VAL_28]], %[[VAL_29]], %[[VAL_32]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_33]]#1 : none
// CHECK:           %[[VAL_34:.*]] = br %[[VAL_32]]#0 : none
// CHECK:           %[[VAL_35:.*]] = br %[[VAL_33]]#0 : i32
// CHECK:           %[[VAL_36:.*]] = mux %[[VAL_13]]#1 {{\[}}%[[VAL_34]], %[[VAL_26]]] : index, none
// CHECK:           %[[VAL_37:.*]] = mux %[[VAL_13]]#0 {{\[}}%[[VAL_35]], %[[VAL_27]]] : index, i32
// CHECK:           return %[[VAL_37]], %[[VAL_36]] : i32, none
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
