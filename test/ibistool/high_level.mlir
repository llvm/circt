// RUN: ibistool --hi --post-ibis-ir %s | FileCheck %s

// CHECK-LABEL:   ibis.class @ToHandshake {
// CHECK:           %[[VAL_0:.*]] = ibis.this @ToHandshake
// CHECK:           ibis.method.df @foo(%[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: none) -> (i32, none) {
// CHECK:             %[[VAL_5:.*]] = handshake.constant %[[VAL_4]] {value = 2 : i32} : i32
// CHECK:             %[[VAL_6:.*]] = handshake.constant %[[VAL_4]] {value = 1 : index} : index
// CHECK:             %[[VAL_7:.*]] = handshake.constant %[[VAL_4]] {value = 0 : i32} : i32
// CHECK:             %[[VAL_8:.*]] = handshake.buffer [1] seq %[[VAL_9:.*]] {initValues = [0]} : i1
// CHECK:             %[[VAL_10:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_4]], %[[VAL_11:.*]]] : i1, none
// CHECK:             %[[VAL_12:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_1]], %[[VAL_13:.*]]] : i1, index
// CHECK:             %[[VAL_14:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_7]], %[[VAL_15:.*]]] : i1, i32
// CHECK:             %[[VAL_16:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_2]], %[[VAL_17:.*]]] : i1, index
// CHECK:             %[[VAL_18:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_5]], %[[VAL_19:.*]]] : i1, i32
// CHECK:             %[[VAL_20:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_6]], %[[VAL_21:.*]]] : i1, index
// CHECK:             %[[VAL_22:.*]] = handshake.mux %[[VAL_8]] {{\[}}%[[VAL_7]], %[[VAL_23:.*]]] : i1, i32
// CHECK:             %[[VAL_9]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_16]] : index
// CHECK:             %[[VAL_24:.*]], %[[VAL_25:.*]] = handshake.cond_br %[[VAL_9]], %[[VAL_12]] : index
// CHECK:             %[[VAL_26:.*]], %[[VAL_27:.*]] = handshake.cond_br %[[VAL_9]], %[[VAL_14]] : i32
// CHECK:             %[[VAL_17]], %[[VAL_28:.*]] = handshake.cond_br %[[VAL_9]], %[[VAL_16]] : index
// CHECK:             %[[VAL_19]], %[[VAL_29:.*]] = handshake.cond_br %[[VAL_9]], %[[VAL_18]] : i32
// CHECK:             %[[VAL_21]], %[[VAL_30:.*]] = handshake.cond_br %[[VAL_9]], %[[VAL_20]] : index
// CHECK:             %[[VAL_23]], %[[VAL_31:.*]] = handshake.cond_br %[[VAL_9]], %[[VAL_22]] : i32
// CHECK:             %[[VAL_32:.*]], %[[VAL_33:.*]] = handshake.cond_br %[[VAL_9]], %[[VAL_10]] : none
// CHECK:             %[[VAL_34:.*]] = arith.index_cast %[[VAL_24]] : index to i32
// CHECK:             %[[VAL_35:.*]] = arith.remsi %[[VAL_26]], %[[VAL_19]] : i32
// CHECK:             %[[VAL_36:.*]] = arith.cmpi eq, %[[VAL_35]], %[[VAL_23]] : i32
// CHECK:             %[[VAL_37:.*]], %[[VAL_38:.*]] = handshake.cond_br %[[VAL_36]], %[[VAL_26]] : i32
// CHECK:             %[[VAL_39:.*]], %[[VAL_40:.*]] = handshake.cond_br %[[VAL_36]], %[[VAL_32]] : none
// CHECK:             %[[VAL_41:.*]], %[[VAL_42:.*]] = handshake.cond_br %[[VAL_36]], %[[VAL_34]] : i32
// CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_37]], %[[VAL_41]] : i32
// CHECK:             %[[VAL_44:.*]] = arith.subi %[[VAL_38]], %[[VAL_42]] : i32
// CHECK:             %[[VAL_15]] = handshake.mux %[[VAL_45:.*]] {{\[}}%[[VAL_44]], %[[VAL_43]]] : index, i32
// CHECK:             %[[VAL_11]], %[[VAL_45]] = handshake.control_merge %[[VAL_40]], %[[VAL_39]] : none, index
// CHECK:             %[[VAL_13]] = arith.addi %[[VAL_24]], %[[VAL_21]] : index
// CHECK:             ibis.return %[[VAL_27]], %[[VAL_33]] : i32, none
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
