// RUN: circt-opt -handshake-insert-buffer=strategies=cycles %s | circt-opt -handshake-insert-buffer=strategies=cycles | FileCheck %s -check-prefix=CHECK


// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = control_merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]]:3 = fork [3] %[[VAL_2]] : none
// CHECK:           sink %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_4]]#1 {value = 1 : index} : index
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_4]]#0 {value = 42 : index} : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_4]]#2 : none
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_6]] : index
// CHECK:           %[[VAL_10:.*]] = mux %[[VAL_11:.*]]#1 {{\[}}%[[VAL_12:.*]], %[[VAL_9]]] : index, index
// CHECK:           %[[VAL_13:.*]] = buffer [2] %[[VAL_10]] {sequential = true} : index
// CHECK:           %[[VAL_14:.*]]:2 = fork [2] %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_17:.*]], %[[VAL_7]] : none
// CHECK:           %[[VAL_18:.*]] = buffer [2] %[[VAL_16]] {sequential = true} : index
// CHECK:           %[[VAL_19:.*]] = buffer [2] %[[VAL_15]] {sequential = true} : none
// CHECK:           %[[VAL_11]]:2 = fork [2] %[[VAL_18]] : index
// CHECK:           %[[VAL_20:.*]] = mux %[[VAL_14]]#0 {{\[}}%[[VAL_21:.*]], %[[VAL_8]]] : index, index
// CHECK:           %[[VAL_22:.*]] = buffer [2] %[[VAL_20]] {sequential = true} : index
// CHECK:           %[[VAL_23:.*]]:2 = fork [2] %[[VAL_22]] : index
// CHECK:           %[[VAL_24:.*]] = arith.cmpi slt, %[[VAL_23]]#1, %[[VAL_14]]#1 : index
// CHECK:           %[[VAL_25:.*]]:3 = fork [3] %[[VAL_24]] : i1
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = cond_br %[[VAL_25]]#2, %[[VAL_14]]#0 : index
// CHECK:           sink %[[VAL_27]] : index
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_25]]#1, %[[VAL_19]] : none
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_25]]#0, %[[VAL_23]]#0 : index
// CHECK:           sink %[[VAL_31]] : index
// CHECK:           %[[VAL_32:.*]] = merge %[[VAL_30]] : index
// CHECK:           %[[VAL_33:.*]] = merge %[[VAL_26]] : index
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = control_merge %[[VAL_28]] : none
// CHECK:           %[[VAL_36:.*]]:2 = fork [2] %[[VAL_34]] : none
// CHECK:           sink %[[VAL_35]] : index
// CHECK:           %[[VAL_37:.*]] = constant %[[VAL_36]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_38:.*]] = arith.addi %[[VAL_32]], %[[VAL_37]] : index
// CHECK:           %[[VAL_12]] = br %[[VAL_33]] : index
// CHECK:           %[[VAL_17]] = br %[[VAL_36]]#1 : none
// CHECK:           %[[VAL_21]] = br %[[VAL_38]] : index
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = control_merge %[[VAL_29]] : none
// CHECK:           sink %[[VAL_40]] : index
// CHECK:           return %[[VAL_39]] : none
// CHECK:         }
module {
  handshake.func @simple_loop(%arg0: none, ...) -> none {
    %0 = br %arg0 : none
    %1:2 = control_merge %0 : none
    %2:3 = fork [3] %1#0 : none
    sink %1#1 : index
    %3 = constant %2#1 {value = 1 : index} : index
    %4 = constant %2#0 {value = 42 : index} : index
    %5 = br %2#2 : none
    %6 = br %3 : index
    %7 = br %4 : index
    %8 = mux %11#1 [%22, %7] : index, index
    %9:2 = fork [2] %8 : index
    %10:2 = control_merge %23, %5 : none  
    %11:2 = fork [2] %10#1 : index
    %12 = mux %9#0 [%24, %6] : index, index
    %13:2 = fork [2] %12 : index
    %14 = arith.cmpi slt, %13#1, %9#1 : index
    %15:3 = fork [3] %14 : i1
    %trueResult, %falseResult = cond_br %15#2, %9#0 : index
    sink %falseResult : index
    %trueResult_0, %falseResult_1 = cond_br %15#1, %10#0 : none
    %trueResult_2, %falseResult_3 = cond_br %15#0, %13#0 : index
    sink %falseResult_3 : index
    %16 = merge %trueResult_2 : index
    %17 = merge %trueResult : index
    %18:2 = control_merge %trueResult_0 : none
    %19:2 = fork [2] %18#0 : none
    sink %18#1 : index
    %20 = constant %19#0 {value = 1 : index} : index
    %21 = arith.addi %16, %20 : index
    %22 = br %17 : index
    %23 = br %19#1 : none
    %24 = br %21 : index
    %25:2 = control_merge %falseResult_1 : none
    sink %25#1 : index
    return %25#0 : none
  }
}
