// RUN: circt-opt -lower-std-to-handshake %s --canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @simpleDiamond(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                  %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]]:3 = fork [3] %[[VAL_0]] : i1
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]]#0 : i1 to index
// CHECK:           %[[VAL_5:.*]] = buffer [2] fifo %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_3]]#2, %[[VAL_1]] : i64
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = cond_br %[[VAL_3]]#1, %[[VAL_2]] : none
// CHECK:           %[[VAL_11:.*]] = mux %[[VAL_6]]#1 {{\[}}%[[VAL_10]], %[[VAL_9]]] : index, none
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_6]]#0 {{\[}}%[[VAL_8]], %[[VAL_7]]] : index, i64
// CHECK:           sink %[[VAL_12]] : i64
// CHECK:           return %[[VAL_11]] : none
// CHECK:         }
func.func @simpleDiamond(%arg0: i1, %arg1: i64) {
  cf.cond_br %arg0, ^bb1(%arg1: i64), ^bb2(%arg1: i64)
^bb1(%v1: i64):  // pred: ^bb0
  cf.br ^bb3(%v1: i64)
^bb2(%v2: i64):  // pred: ^bb0
  cf.br ^bb3(%v2: i64)
^bb3(%v3: i64):  // 2 preds: ^bb1, ^bb2
  return
}

// -----

// CHECK-LABEL:   handshake.func @nestedDiamond(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:4 = fork [4] %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]]#0 : i1 to index
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_2]]#2, %[[VAL_2]]#3 : i1
// CHECK:           sink %[[VAL_6]] : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_1]] : none
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_5]] : i1
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]]#0 : i1 to index
// CHECK:           %[[VAL_11:.*]] = buffer [2] fifo %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_9]]#1, %[[VAL_7]] : none
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_11]] {{\[}}%[[VAL_13]], %[[VAL_12]]] : index, none
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_14]], %[[VAL_8]]] : index, none
// CHECK:           return %[[VAL_15]] : none
// CHECK:         }
func.func @nestedDiamond(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb5
^bb3:  // pred: ^bb1
  cf.br ^bb5
^bb4:  // pred: ^bb0
  cf.br ^bb6
^bb5:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb6
^bb6:  // 2 preds: ^bb4, ^bb5
  return
}

// -----

// CHECK-LABEL:   handshake.func @triangle(
// CHECK-SAME:                             %[[VAL_0:.*]]: i1,
// CHECK-SAME:                             %[[VAL_1:.*]]: i64,
// CHECK-SAME:                             %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]]:3 = fork [3] %[[VAL_0]] : i1
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]]#0 : i1 to index
// CHECK:           %[[VAL_5:.*]] = buffer [2] fifo %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_3]]#2, %[[VAL_1]] : i64
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = cond_br %[[VAL_3]]#1, %[[VAL_2]] : none
// CHECK:           %[[VAL_11:.*]] = mux %[[VAL_6]]#1 {{\[}}%[[VAL_9]], %[[VAL_10]]] : index, none
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_6]]#0 {{\[}}%[[VAL_7]], %[[VAL_8]]] : index, i64
// CHECK:           sink %[[VAL_12]] : i64
// CHECK:           return %[[VAL_11]] : none
// CHECK:         }
func.func @triangle(%arg0: i1, %val0: i64) {
  cf.cond_br %arg0, ^bb1(%val0: i64), ^bb2(%val0: i64)
^bb1(%val1: i64):  // pred: ^bb0
  cf.br ^bb2(%val1: i64)
^bb2(%val2: i64):  // 2 preds: ^bb0, ^bb1
  return
}

// -----

// CHECK-LABEL:   handshake.func @nestedTriangle(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                   %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:4 = fork [4] %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]]#0 : i1 to index
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_2]]#2, %[[VAL_2]]#3 : i1
// CHECK:           sink %[[VAL_6]] : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_1]] : none
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_5]] : i1
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]]#0 : i1 to index
// CHECK:           %[[VAL_11:.*]] = buffer [2] fifo %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_9]]#1, %[[VAL_7]] : none
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_11]] {{\[}}%[[VAL_12]], %[[VAL_13]]] : index, none
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_14]], %[[VAL_8]]] : index, none
// CHECK:           return %[[VAL_15]] : none
// CHECK:         }
func.func @nestedTriangle(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // 2 preds: ^bb0, ^bb3
  return
}

// -----

// CHECK-LABEL:   handshake.func @multiple_blocks_needed(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                           %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:4 = fork [4] %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]]#0 : i1 to index
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]]#2, %[[VAL_2]]#3 : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_1]] : none
// CHECK:           %[[VAL_10:.*]]:4 = fork [4] %[[VAL_6]] : i1
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]]#0 : i1 to index
// CHECK:           %[[VAL_12:.*]] = buffer [2] fifo %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]]:2 = fork [2] %[[VAL_12]] : index
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_10]]#2, %[[VAL_10]]#3 : i1
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = cond_br %[[VAL_10]]#1, %[[VAL_8]] : none
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_13]]#0 {{\[}}%[[VAL_14]], %[[VAL_15]]] : index, i1
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_13]]#1 {{\[}}%[[VAL_16]], %[[VAL_17]]] : index, none
// CHECK:           %[[VAL_20:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_18]], %[[VAL_7]]] : index, i1
// CHECK:           %[[VAL_21:.*]]:4 = fork [4] %[[VAL_20]] : i1
// CHECK:           %[[VAL_22:.*]] = arith.index_cast %[[VAL_21]]#0 : i1 to index
// CHECK:           %[[VAL_23:.*]] = buffer [2] fifo %[[VAL_22]] : index
// CHECK:           %[[VAL_24:.*]] = mux %[[VAL_5]]#1 {{\[}}%[[VAL_19]], %[[VAL_9]]] : index, none
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = cond_br %[[VAL_21]]#2, %[[VAL_21]]#3 : i1
// CHECK:           sink %[[VAL_26]] : i1
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = cond_br %[[VAL_21]]#1, %[[VAL_24]] : none
// CHECK:           %[[VAL_29:.*]]:2 = fork [2] %[[VAL_25]] : i1
// CHECK:           %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]]#0 : i1 to index
// CHECK:           %[[VAL_31:.*]] = buffer [2] fifo %[[VAL_30]] : index
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_29]]#1, %[[VAL_27]] : none
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_31]] {{\[}}%[[VAL_32]], %[[VAL_33]]] : index, none
// CHECK:           %[[VAL_35:.*]] = mux %[[VAL_23]] {{\[}}%[[VAL_34]], %[[VAL_28]]] : index, none
// CHECK:           return %[[VAL_35]] : none
// CHECK:         }
func.func @multiple_blocks_needed(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // 2 preds: ^bb0, ^bb3
  cf.cond_br %arg0, ^bb5, ^bb8
^bb5:  // pred: ^bb4
  cf.cond_br %arg0, ^bb6, ^bb7
^bb6:  // pred: ^bb5
  cf.br ^bb7
^bb7:  // 2 preds: ^bb5, ^bb6
  cf.br ^bb8
^bb8:  // 2 preds: ^bb4, ^bb7
  return
}

// -----

// CHECK-LABEL:   handshake.func @sameSuccessor(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_3]]#1 : none
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_4]] : none
// CHECK:           sink %[[VAL_5]] : none
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_2]]#0, %[[VAL_3]]#0 : none
// CHECK:           sink %[[VAL_8]] : none
// CHECK:           sink %[[VAL_7]] : none
// CHECK:           %[[VAL_9:.*]] = merge %[[VAL_6]]#0, %[[VAL_6]]#1 : none
// CHECK:           return %[[VAL_9]] : none
// CHECK:         }
func.func @sameSuccessor(%cond: i1) {
  cf.cond_br %cond, ^1, ^1
^1:
  return
}

// -----

// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]]#0 {value = 1 : i64} : i64
// CHECK:           %[[VAL_4:.*]] = buffer [1] seq %[[VAL_5:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_6:.*]]:3 = fork [3] %[[VAL_4]] : i1
// CHECK:           %[[VAL_7:.*]] = mux %[[VAL_6]]#2 {{\[}}%[[VAL_2]]#1, %[[VAL_8:.*]]#1] : i1, none
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_7]] : none
// CHECK:           %[[VAL_10:.*]] = mux %[[VAL_6]]#1 {{\[}}%[[VAL_0]], %[[VAL_11:.*]]] : i1, i64
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_10]] : i64
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_6]]#0 {{\[}}%[[VAL_3]], %[[VAL_14:.*]]] : i1, i64
// CHECK:           %[[VAL_15:.*]]:2 = fork [2] %[[VAL_13]] : i64
// CHECK:           %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]]#0, %[[VAL_12]]#0 : i64
// CHECK:           %[[VAL_17:.*]]:4 = fork [4] %[[VAL_16]] : i1
// CHECK:           %[[VAL_18:.*]], %[[VAL_11]] = cond_br %[[VAL_17]]#3, %[[VAL_12]]#1 : i64
// CHECK:           sink %[[VAL_18]] : i64
// CHECK:           %[[VAL_19:.*]] = constant %[[VAL_9]]#0 {value = true} : i1
// CHECK:           %[[VAL_5]] = arith.xori %[[VAL_17]]#0, %[[VAL_19]] : i1
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = cond_br %[[VAL_17]]#2, %[[VAL_9]]#1 : none
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = cond_br %[[VAL_17]]#1, %[[VAL_15]]#1 : i64
// CHECK:           sink %[[VAL_22]] : i64
// CHECK:           %[[VAL_8]]:2 = fork [2] %[[VAL_21]] : none
// CHECK:           %[[VAL_24:.*]] = constant %[[VAL_8]]#0 {value = 1 : i64} : i64
// CHECK:           %[[VAL_14]] = arith.addi %[[VAL_23]], %[[VAL_24]] : i64
// CHECK:           return %[[VAL_20]] : none
// CHECK:         }
func.func @simple_loop(%arg0: i64) {
  %c1_i64 = arith.constant 1 : i64
  cf.br ^bb1(%c1_i64 : i64)
^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
  %1 = arith.cmpi eq, %0, %arg0 : i64
  cf.cond_br %1, ^bb3, ^bb2
^bb2:  // pred: ^bb1
  %c1_i64_0 = arith.constant 1 : i64
  %2 = arith.addi %0, %c1_i64_0 : i64
  cf.br ^bb1(%2 : i64)
^bb3:  // pred: ^bb1
  return
}

// -----

// CHECK-LABEL:   handshake.func @blockWith3PredsAndLoop(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                           %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:4 = fork [4] %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]]#0 : i1 to index
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_2]]#2, %[[VAL_2]]#3 : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_1]] : none
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_5]] : i1
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]]#0 : i1 to index
// CHECK:           %[[VAL_11:.*]] = buffer [2] fifo %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_9]]#1, %[[VAL_7]] : none
// CHECK:           %[[VAL_14:.*]] = buffer [1] seq %[[VAL_15:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_16:.*]]:2 = fork [2] %[[VAL_14]] : i1
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_16]]#1 {{\[}}%[[VAL_8]], %[[VAL_18:.*]]] : i1, none
// CHECK:           %[[VAL_19:.*]]:2 = fork [2] %[[VAL_17]] : none
// CHECK:           %[[VAL_20:.*]] = mux %[[VAL_16]]#0 {{\[}}%[[VAL_6]], %[[VAL_21:.*]]] : i1, i1
// CHECK:           %[[VAL_22:.*]]:4 = fork [4] %[[VAL_20]] : i1
// CHECK:           %[[VAL_23:.*]], %[[VAL_21]] = cond_br %[[VAL_22]]#0, %[[VAL_22]]#1 : i1
// CHECK:           sink %[[VAL_23]] : i1
// CHECK:           %[[VAL_24:.*]] = constant %[[VAL_19]]#0 {value = true} : i1
// CHECK:           %[[VAL_15]] = arith.xori %[[VAL_22]]#3, %[[VAL_24]] : i1
// CHECK:           %[[VAL_25:.*]], %[[VAL_18]] = cond_br %[[VAL_22]]#2, %[[VAL_19]]#1 : none
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_11]] {{\[}}%[[VAL_13]], %[[VAL_12]]] : index, none
// CHECK:           %[[VAL_27:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_26]], %[[VAL_25]]] : index, none
// CHECK:           return %[[VAL_27]] : none
// CHECK:         }
func.func @blockWith3PredsAndLoop(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb7
^bb3:  // pred: ^bb1
  cf.br ^bb7
^bb4:  // pred: ^bb0
  cf.br ^bb5
^bb5:  // 2 preds: ^bb4, ^bb6
  cf.cond_br %arg0, ^bb8, ^bb6
^bb6:  // pred: ^bb5
  cf.br ^bb5
^bb7:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb8
^bb8:  // 2 preds: ^bb5, ^bb7
  return
}

// -----

// CHECK-LABEL:   handshake.func @otherBlockOrder(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                    %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:4 = fork [4] %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]]#0 : i1 to index
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_2]]#2, %[[VAL_2]]#3 : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_1]] : none
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_5]] : i1
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]]#0 : i1 to index
// CHECK:           %[[VAL_11:.*]] = buffer [2] fifo %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_9]]#1, %[[VAL_7]] : none
// CHECK:           %[[VAL_14:.*]] = buffer [1] seq %[[VAL_15:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_16:.*]]:2 = fork [2] %[[VAL_14]] : i1
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_16]]#1 {{\[}}%[[VAL_8]], %[[VAL_18:.*]]] : i1, none
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_16]]#0 {{\[}}%[[VAL_6]], %[[VAL_20:.*]]] : i1, i1
// CHECK:           %[[VAL_21:.*]]:4 = fork [4] %[[VAL_19]] : i1
// CHECK:           %[[VAL_22:.*]]:2 = fork [2] %[[VAL_17]] : none
// CHECK:           %[[VAL_23:.*]], %[[VAL_20]] = cond_br %[[VAL_21]]#2, %[[VAL_21]]#3 : i1
// CHECK:           sink %[[VAL_23]] : i1
// CHECK:           %[[VAL_24:.*]] = constant %[[VAL_22]]#0 {value = true} : i1
// CHECK:           %[[VAL_15]] = arith.xori %[[VAL_21]]#0, %[[VAL_24]] : i1
// CHECK:           %[[VAL_25:.*]], %[[VAL_18]] = cond_br %[[VAL_21]]#1, %[[VAL_22]]#1 : none
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_11]] {{\[}}%[[VAL_13]], %[[VAL_12]]] : index, none
// CHECK:           %[[VAL_27:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_26]], %[[VAL_25]]] : index, none
// CHECK:           return %[[VAL_27]] : none
// CHECK:         }
func.func @otherBlockOrder(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb7
^bb3:  // pred: ^bb1
  cf.br ^bb7
^bb4:  // pred: ^bb0
  cf.br ^bb5
^bb5:  // 2 preds: ^bb4, ^bb6
  cf.br ^bb6
^bb6:  // pred: ^bb5
  cf.cond_br %arg0, ^bb8, ^bb5
^bb7:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb8
^bb8:  // 2 preds: ^bb6, ^bb7
  return
}

// -----

// CHECK-LABEL:   handshake.func @multiple_block_args(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                        %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                        %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]]:6 = fork [6] %[[VAL_0]] : i1
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]]#0 : i1 to index
// CHECK:           %[[VAL_5:.*]] = buffer [2] fifo %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_1]] : i64
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_3]]#4, %[[VAL_3]]#5 : i1
// CHECK:           sink %[[VAL_8]] : i1
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = cond_br %[[VAL_3]]#3, %[[VAL_6]]#1 : i64
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = cond_br %[[VAL_3]]#2, %[[VAL_6]]#0 : i64
// CHECK:           sink %[[VAL_11]] : i64
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_3]]#1, %[[VAL_2]] : none
// CHECK:           %[[VAL_15:.*]]:4 = fork [4] %[[VAL_7]] : i1
// CHECK:           %[[VAL_16:.*]] = arith.index_cast %[[VAL_15]]#0 : i1 to index
// CHECK:           %[[VAL_17:.*]] = buffer [2] fifo %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]]:2 = fork [2] %[[VAL_9]] : i64
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = cond_br %[[VAL_15]]#3, %[[VAL_13]] : none
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = cond_br %[[VAL_15]]#2, %[[VAL_18]]#1 : i64
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = cond_br %[[VAL_15]]#1, %[[VAL_18]]#0 : i64
// CHECK:           sink %[[VAL_23]] : i64
// CHECK:           sink %[[VAL_21]] : i64
// CHECK:           sink %[[VAL_24]] : i64
// CHECK:           sink %[[VAL_22]] : i64
// CHECK:           sink %[[VAL_12]] : i64
// CHECK:           sink %[[VAL_10]] : i64
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_17]] {{\[}}%[[VAL_20]], %[[VAL_19]]] : index, none
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_5]] {{\[}}%[[VAL_25]], %[[VAL_14]]] : index, none
// CHECK:           return %[[VAL_26]] : none
// CHECK:         }
func.func @multiple_block_args(%arg0: i1, %arg1: i64) {
  cf.cond_br %arg0, ^bb1(%arg1 : i64), ^bb4(%arg1, %arg1 : i64, i64)
^bb1(%0: i64):  // pred: ^bb0
  cf.cond_br %arg0, ^bb2(%0 : i64), ^bb3(%0, %0 : i64, i64)
^bb2(%1: i64):  // pred: ^bb1
  cf.br ^bb5
^bb3(%2: i64, %3: i64):  // pred: ^bb1
  cf.br ^bb5
^bb4(%4: i64, %5: i64):  // pred: ^bb0
  cf.br ^bb6
^bb5:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb6
^bb6:  // 2 preds: ^bb4, ^bb5
  return
}

// -----

// CHECK-LABEL:   handshake.func @mergeBlockAsLoopHeader(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                           %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]]:4 = fork [4] %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]]#0 : i1 to index
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]]#2, %[[VAL_2]]#3 : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_1]] : none
// CHECK:           %[[VAL_10:.*]] = mux %[[VAL_5]]#1 {{\[}}%[[VAL_9]], %[[VAL_8]]] : index, none
// CHECK:           %[[VAL_11:.*]]:2 = fork [2] %[[VAL_10]] : none
// CHECK:           %[[VAL_12:.*]] = buffer [1] seq %[[VAL_13:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_14:.*]]:2 = fork [2] %[[VAL_12]] : i1
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_14]]#1 {{\[}}%[[VAL_11]]#1, %[[VAL_16:.*]]] : i1, none
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_7]], %[[VAL_6]]] : index, i1
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_14]]#0 {{\[}}%[[VAL_17]], %[[VAL_19:.*]]] : i1, i1
// CHECK:           %[[VAL_20:.*]]:4 = fork [4] %[[VAL_18]] : i1
// CHECK:           %[[VAL_21:.*]], %[[VAL_19]] = cond_br %[[VAL_20]]#0, %[[VAL_20]]#1 : i1
// CHECK:           sink %[[VAL_21]] : i1
// CHECK:           %[[VAL_22:.*]] = constant %[[VAL_11]]#0 {value = true} : i1
// CHECK:           %[[VAL_13]] = arith.xori %[[VAL_20]]#3, %[[VAL_22]] : i1
// CHECK:           %[[VAL_23:.*]], %[[VAL_16]] = cond_br %[[VAL_20]]#2, %[[VAL_15]] : none
// CHECK:           return %[[VAL_23]] : none
// CHECK:         }
func.func @mergeBlockAsLoopHeader(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb3
^bb2:  // pred: ^bb0
  cf.br ^bb3
^bb3:  // pred: ^bb1, ^bb2, ^bb4
  cf.cond_br %arg0, ^bb5, ^bb4
^bb4:  // pred: ^bb1, ^bb2
  cf.br ^bb3
^bb5:  // 2 preds: ^bb3
  return
}

