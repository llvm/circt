// RUN: circt-opt -lower-std-to-handshake -split-input-file %s | FileCheck %s

// CHECK-LABEL:   handshake.func @bar(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i32
// CHECK:           return %[[VAL_2]], %[[VAL_1]] : i32, none
// CHECK:         }
func.func @bar(%0 : i32) -> i32 {
  return %0 : i32
}

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none)
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
// CHECK:           %[[VAL_8:.*]] = buffer [2] fifo %[[VAL_7]]#0 : i1
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_7]]#3, %[[VAL_4]] : i32
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_7]]#2, %[[VAL_5]] : i32
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_7]]#1, %[[VAL_3]] : none
// CHECK:           %[[VAL_16:.*]] = merge %[[VAL_10]] : i32
// CHECK:           %[[VAL_17:.*]] = merge %[[VAL_12]] : i32
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = control_merge %[[VAL_14]] : none
// CHECK:           %[[VAL_20:.*]]:2 = fork [2] %[[VAL_18]] : none
// CHECK:           sink %[[VAL_19]] : index
// CHECK:           %[[VAL_21:.*]]:2 = instance @add(%[[VAL_16]], %[[VAL_17]], %[[VAL_20]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_21]]#1 : none
// CHECK:           %[[VAL_22:.*]] = br %[[VAL_20]]#0 : none
// CHECK:           %[[VAL_23:.*]] = br %[[VAL_21]]#0 : i32
// CHECK:           %[[VAL_24:.*]] = merge %[[VAL_11]] : i32
// CHECK:           %[[VAL_25:.*]] = merge %[[VAL_13]] : i32
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = control_merge %[[VAL_15]] : none
// CHECK:           %[[VAL_28:.*]]:2 = fork [2] %[[VAL_26]] : none
// CHECK:           sink %[[VAL_27]] : index
// CHECK:           %[[VAL_29:.*]]:2 = instance @sub(%[[VAL_24]], %[[VAL_25]], %[[VAL_28]]#1) : (i32, i32, none) -> (i32, none)
// CHECK:           sink %[[VAL_29]]#1 : none
// CHECK:           %[[VAL_30:.*]] = br %[[VAL_28]]#0 : none
// CHECK:           %[[VAL_31:.*]] = br %[[VAL_29]]#0 : i32
// CHECK:           %[[VAL_32:.*]] = mux %[[VAL_9]]#1 {{\[}}%[[VAL_30]], %[[VAL_22]]] : i1, none
// CHECK:           %[[VAL_33:.*]] = arith.index_cast %[[VAL_9]]#0 : i1 to index
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_33]] {{\[}}%[[VAL_31]], %[[VAL_23]]] : index, i32
// CHECK:           return %[[VAL_34]], %[[VAL_32]] : i32, none
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
