// RUN: ibistool --hi --post-ibis-ir %s | FileCheck %s

// CHECK-LABEL:   ibis.class @ToHandshake {
// CHECK:           %[[VAL_0:.*]] = ibis.this @ToHandshake
// CHECK:           ibis.method.df @foo(%[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: none) -> (i32, none) {
// CHECK:             %[[VAL_5:.*]] = handshake.constant %[[VAL_4]] {value = 0 : i32} : i32
// CHECK:             %[[VAL_6:.*]] = handshake.constant %[[VAL_4]] {value = 1 : index} : index
// CHECK:             %[[VAL_7:.*]] = handshake.constant %[[VAL_4]] {value = 2 : i32} : i32
// CHECK:             %[[VAL_8:.*]]:3 = ibis.sblock () -> (i32, index, i32) {
// CHECK:               ibis.sblock.return %[[VAL_7]], %[[VAL_6]], %[[VAL_5]] : i32, index, i32
// CHECK:             }
// CHECK:             %[[VAL_9:.*]] = handshake.buffer [1] seq %[[VAL_10:.*]] {initValues = [0]} : i1
// CHECK:             %[[VAL_11:.*]] = handshake.mux %[[VAL_9]] {{\[}}%[[VAL_4]], %[[VAL_12:.*]]] : i1, none
// CHECK:             %[[VAL_13:.*]] = handshake.mux %[[VAL_9]] {{\[}}%[[VAL_1]], %[[VAL_14:.*]]] : i1, index
// CHECK:             %[[VAL_15:.*]] = handshake.mux %[[VAL_9]] {{\[}}%[[VAL_16:.*]]#2, %[[VAL_17:.*]]] : i1, i32
// CHECK:             %[[VAL_18:.*]] = handshake.mux %[[VAL_9]] {{\[}}%[[VAL_2]], %[[VAL_19:.*]]] : i1, index
// CHECK:             %[[VAL_20:.*]] = handshake.mux %[[VAL_9]] {{\[}}%[[VAL_16]]#0, %[[VAL_21:.*]]] : i1, i32
// CHECK:             %[[VAL_22:.*]] = handshake.mux %[[VAL_9]] {{\[}}%[[VAL_16]]#1, %[[VAL_23:.*]]] : i1, index
// CHECK:             %[[VAL_24:.*]] = handshake.mux %[[VAL_9]] {{\[}}%[[VAL_16]]#2, %[[VAL_25:.*]]] : i1, i32
// CHECK:             %[[VAL_10]] = ibis.sblock (%[[VAL_26:.*]] : index = %[[VAL_13]], %[[VAL_27:.*]] : index = %[[VAL_18]]) -> i1 {
// CHECK:               %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_27]] : index
// CHECK:               ibis.sblock.return %[[VAL_28]] : i1
// CHECK:             }
// CHECK:             %[[VAL_29:.*]], %[[VAL_30:.*]] = handshake.cond_br %[[VAL_10]], %[[VAL_13]] : index
// CHECK:             %[[VAL_31:.*]], %[[VAL_32:.*]] = handshake.cond_br %[[VAL_10]], %[[VAL_15]] : i32
// CHECK:             %[[VAL_19]], %[[VAL_33:.*]] = handshake.cond_br %[[VAL_10]], %[[VAL_18]] : index
// CHECK:             %[[VAL_21]], %[[VAL_34:.*]] = handshake.cond_br %[[VAL_10]], %[[VAL_20]] : i32
// CHECK:             %[[VAL_23]], %[[VAL_35:.*]] = handshake.cond_br %[[VAL_10]], %[[VAL_22]] : index
// CHECK:             %[[VAL_25]], %[[VAL_36:.*]] = handshake.cond_br %[[VAL_10]], %[[VAL_24]] : i32
// CHECK:             %[[VAL_37:.*]], %[[VAL_38:.*]] = handshake.cond_br %[[VAL_10]], %[[VAL_11]] : none
// CHECK:             %[[VAL_39:.*]]:2 = ibis.sblock (%[[VAL_26]] : index = %[[VAL_29]], %[[VAL_40:.*]] : i32 = %[[VAL_31]], %[[VAL_41:.*]] : i32 = %[[VAL_21]], %[[VAL_42:.*]] : i32 = %[[VAL_25]]) -> (i32, i1) {
// CHECK:               %[[VAL_43:.*]] = arith.index_cast %[[VAL_26]] : index to i32
// CHECK:               %[[VAL_44:.*]] = arith.remsi %[[VAL_40]], %[[VAL_41]] : i32
// CHECK:               %[[VAL_45:.*]] = arith.cmpi eq, %[[VAL_44]], %[[VAL_42]] : i32
// CHECK:               ibis.sblock.return %[[VAL_43]], %[[VAL_45]] : i32, i1
// CHECK:             }
// CHECK:             %[[VAL_46:.*]], %[[VAL_47:.*]] = handshake.cond_br %[[VAL_48:.*]]#1, %[[VAL_31]] : i32
// CHECK:             %[[VAL_49:.*]], %[[VAL_50:.*]] = handshake.cond_br %[[VAL_48]]#1, %[[VAL_37]] : none
// CHECK:             %[[VAL_51:.*]], %[[VAL_52:.*]] = handshake.cond_br %[[VAL_48]]#1, %[[VAL_48]]#0 : i32
// CHECK:             %[[VAL_53:.*]] = ibis.sblock (%[[VAL_26]] : i32 = %[[VAL_46]], %[[VAL_54:.*]] : i32 = %[[VAL_51]]) -> i32 {
// CHECK:               %[[VAL_55:.*]] = arith.addi %[[VAL_26]], %[[VAL_54]] : i32
// CHECK:               ibis.sblock.return %[[VAL_55]] : i32
// CHECK:             }
// CHECK:             %[[VAL_56:.*]] = ibis.sblock (%[[VAL_26]] : i32 = %[[VAL_47]], %[[VAL_57:.*]] : i32 = %[[VAL_52]]) -> i32 {
// CHECK:               %[[VAL_58:.*]] = arith.subi %[[VAL_26]], %[[VAL_57]] : i32
// CHECK:               ibis.sblock.return %[[VAL_58]] : i32
// CHECK:             }
// CHECK:             %[[VAL_17]] = handshake.mux %[[VAL_59:.*]] {{\[}}%[[VAL_56]], %[[VAL_53]]] : index, i32
// CHECK:             %[[VAL_12]], %[[VAL_59]] = handshake.control_merge %[[VAL_50]], %[[VAL_49]] : none, index
// CHECK:             %[[VAL_14]] = ibis.sblock (%[[VAL_26]] : index = %[[VAL_29]], %[[VAL_60:.*]] : index = %[[VAL_23]]) -> index {
// CHECK:               %[[VAL_61:.*]] = arith.addi %[[VAL_26]], %[[VAL_60]] : index
// CHECK:               ibis.sblock.return %[[VAL_61]] : index
// CHECK:             }
// CHECK:             ibis.return %[[VAL_32]], %[[VAL_38]] : i32, none
// CHECK:           }
// CHECK:         }

ibis.class @ToHandshake {
  %this = ibis.this @ToHandshake
  ibis.method @foo(%a: index, %b: index, %c : i1) -> i32 {
    %sum = memref.alloca () : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    memref.store %c0_i32, %sum[] : memref<i32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    scf.for %i = %a to %b step %c1 {
      %acc = memref.load %sum[] : memref<i32>
      %i_i32 = arith.index_cast %i : index to i32
            %rem = arith.remsi %acc, %c2 : i32
      %cond = arith.cmpi eq, %rem, %c0 : i32
      %res = scf.if %cond -> (i32) {
        %v = arith.addi %acc, %i_i32 : i32
        scf.yield %v : i32
      } else {
        %v = arith.subi %acc, %i_i32 : i32
        scf.yield %v : i32
      }
      memref.store %res, %sum[] : memref<i32>
    }
    %res = memref.load %sum[] : memref<i32>
    ibis.return %res : i32
  }
}
