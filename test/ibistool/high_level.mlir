// RUN: ibistool --hi --post-ibis-ir %s | FileCheck %s

// CHECK-LABEL:   ibis.class @ToHandshake {
// CHECK:           %[[VAL_0:.*]] = ibis.this @ToHandshake
// CHECK:           ibis.method.df @foo(%[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: i1, %[[VAL_4:.*]]: none) -> (i32, none) {
// CHECK:             %[[VAL_5:.*]] = handshake.constant %[[VAL_4]] {value = 0 : i32} : i32
// CHECK:             %[[VAL_6:.*]] = handshake.constant %[[VAL_4]] {value = 1 : index} : index
// CHECK:             %[[VAL_7:.*]] = handshake.constant %[[VAL_4]] {value = 2 : i32} : i32
// CHECK:             %[[VAL_8:.*]]:3 = ibis.sblock () -> (i32, index, i32) {
// CHECK:               %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]], %[[VAL_12:.*]] = ibis.pipeline.header
// CHECK:               %[[VAL_13:.*]], %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]] = pipeline.unscheduled() stall(%[[VAL_12]]) clock(%[[VAL_9]]) reset(%[[VAL_10]]) go(%[[VAL_11]]) entryEn(%[[VAL_17:.*]])  -> (out0 : i32, out1 : index, out2 : i32) {
// CHECK:                 pipeline.return %[[VAL_7]], %[[VAL_6]], %[[VAL_5]] : i32, index, i32
// CHECK:               }
// CHECK:               ibis.sblock.return %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]] : i32, index, i32
// CHECK:             }
// CHECK:             %[[VAL_21:.*]] = handshake.buffer [1] seq %[[VAL_22:.*]] {initValues = [0]} : i1
// CHECK:             %[[VAL_23:.*]] = handshake.mux %[[VAL_21]] {{\[}}%[[VAL_4]], %[[VAL_24:.*]]] : i1, none
// CHECK:             %[[VAL_25:.*]] = handshake.mux %[[VAL_21]] {{\[}}%[[VAL_1]], %[[VAL_26:.*]]] : i1, index
// CHECK:             %[[VAL_27:.*]] = handshake.mux %[[VAL_21]] {{\[}}%[[VAL_28:.*]]#2, %[[VAL_29:.*]]] : i1, i32
// CHECK:             %[[VAL_30:.*]] = handshake.mux %[[VAL_21]] {{\[}}%[[VAL_2]], %[[VAL_31:.*]]] : i1, index
// CHECK:             %[[VAL_32:.*]] = handshake.mux %[[VAL_21]] {{\[}}%[[VAL_28]]#0, %[[VAL_33:.*]]] : i1, i32
// CHECK:             %[[VAL_34:.*]] = handshake.mux %[[VAL_21]] {{\[}}%[[VAL_28]]#1, %[[VAL_35:.*]]] : i1, index
// CHECK:             %[[VAL_36:.*]] = handshake.mux %[[VAL_21]] {{\[}}%[[VAL_28]]#2, %[[VAL_37:.*]]] : i1, i32
// CHECK:             %[[VAL_22]] = ibis.sblock (%[[VAL_38:.*]] : index = %[[VAL_25]], %[[VAL_39:.*]] : index = %[[VAL_30]]) -> i1 {
// CHECK:               %[[VAL_40:.*]], %[[VAL_41:.*]], %[[VAL_42:.*]], %[[VAL_43:.*]] = ibis.pipeline.header
// CHECK:               %[[VAL_44:.*]], %[[VAL_45:.*]] = pipeline.unscheduled(%[[VAL_46:.*]] : index = %[[VAL_38]], %[[VAL_47:.*]] : index = %[[VAL_39]]) stall(%[[VAL_43]]) clock(%[[VAL_40]]) reset(%[[VAL_41]]) go(%[[VAL_42]]) entryEn(%[[VAL_48:.*]])  -> (out0 : i1) {
// CHECK:                 %[[VAL_49:.*]] = arith.cmpi slt, %[[VAL_46]], %[[VAL_47]] : index
// CHECK:                 pipeline.return %[[VAL_49]] : i1
// CHECK:               }
// CHECK:               ibis.sblock.return %[[VAL_50:.*]] : i1
// CHECK:             }
// CHECK:             %[[VAL_51:.*]], %[[VAL_52:.*]] = handshake.cond_br %[[VAL_22]], %[[VAL_25]] : index
// CHECK:             %[[VAL_53:.*]], %[[VAL_54:.*]] = handshake.cond_br %[[VAL_22]], %[[VAL_27]] : i32
// CHECK:             %[[VAL_31]], %[[VAL_55:.*]] = handshake.cond_br %[[VAL_22]], %[[VAL_30]] : index
// CHECK:             %[[VAL_33]], %[[VAL_56:.*]] = handshake.cond_br %[[VAL_22]], %[[VAL_32]] : i32
// CHECK:             %[[VAL_35]], %[[VAL_57:.*]] = handshake.cond_br %[[VAL_22]], %[[VAL_34]] : index
// CHECK:             %[[VAL_37]], %[[VAL_58:.*]] = handshake.cond_br %[[VAL_22]], %[[VAL_36]] : i32
// CHECK:             %[[VAL_59:.*]], %[[VAL_60:.*]] = handshake.cond_br %[[VAL_22]], %[[VAL_23]] : none
// CHECK:             %[[VAL_61:.*]]:2 = ibis.sblock (%[[VAL_38]] : index = %[[VAL_51]], %[[VAL_62:.*]] : i32 = %[[VAL_53]], %[[VAL_63:.*]] : i32 = %[[VAL_33]], %[[VAL_64:.*]] : i32 = %[[VAL_37]]) -> (i32, i1) {
// CHECK:               %[[VAL_65:.*]], %[[VAL_66:.*]], %[[VAL_67:.*]], %[[VAL_68:.*]] = ibis.pipeline.header
// CHECK:               %[[VAL_69:.*]], %[[VAL_70:.*]], %[[VAL_71:.*]] = pipeline.unscheduled(%[[VAL_72:.*]] : index = %[[VAL_38]], %[[VAL_73:.*]] : i32 = %[[VAL_62]], %[[VAL_74:.*]] : i32 = %[[VAL_63]], %[[VAL_75:.*]] : i32 = %[[VAL_64]]) stall(%[[VAL_68]]) clock(%[[VAL_65]]) reset(%[[VAL_66]]) go(%[[VAL_67]]) entryEn(%[[VAL_76:.*]])  -> (out0 : i32, out1 : i1) {
// CHECK:                 %[[VAL_77:.*]] = arith.index_cast %[[VAL_72]] : index to i32
// CHECK:                 %[[VAL_78:.*]] = arith.remsi %[[VAL_73]], %[[VAL_74]] : i32
// CHECK:                 %[[VAL_79:.*]] = arith.cmpi eq, %[[VAL_78]], %[[VAL_75]] : i32
// CHECK:                 pipeline.return %[[VAL_77]], %[[VAL_79]] : i32, i1
// CHECK:               }
// CHECK:               ibis.sblock.return %[[VAL_80:.*]], %[[VAL_81:.*]] : i32, i1
// CHECK:             }
// CHECK:             %[[VAL_82:.*]], %[[VAL_83:.*]] = handshake.cond_br %[[VAL_84:.*]]#1, %[[VAL_53]] : i32
// CHECK:             %[[VAL_85:.*]], %[[VAL_86:.*]] = handshake.cond_br %[[VAL_84]]#1, %[[VAL_59]] : none
// CHECK:             %[[VAL_87:.*]], %[[VAL_88:.*]] = handshake.cond_br %[[VAL_84]]#1, %[[VAL_84]]#0 : i32
// CHECK:             %[[VAL_89:.*]] = ibis.sblock (%[[VAL_38]] : i32 = %[[VAL_82]], %[[VAL_90:.*]] : i32 = %[[VAL_87]]) -> i32 {
// CHECK:               %[[VAL_91:.*]], %[[VAL_92:.*]], %[[VAL_93:.*]], %[[VAL_94:.*]] = ibis.pipeline.header
// CHECK:               %[[VAL_95:.*]], %[[VAL_96:.*]] = pipeline.unscheduled(%[[VAL_97:.*]] : i32 = %[[VAL_38]], %[[VAL_98:.*]] : i32 = %[[VAL_90]]) stall(%[[VAL_94]]) clock(%[[VAL_91]]) reset(%[[VAL_92]]) go(%[[VAL_93]]) entryEn(%[[VAL_99:.*]])  -> (out0 : i32) {
// CHECK:                 %[[VAL_100:.*]] = arith.addi %[[VAL_97]], %[[VAL_98]] : i32
// CHECK:                 pipeline.return %[[VAL_100]] : i32
// CHECK:               }
// CHECK:               ibis.sblock.return %[[VAL_101:.*]] : i32
// CHECK:             }
// CHECK:             %[[VAL_102:.*]] = ibis.sblock (%[[VAL_38]] : i32 = %[[VAL_83]], %[[VAL_103:.*]] : i32 = %[[VAL_88]]) -> i32 {
// CHECK:               %[[VAL_104:.*]], %[[VAL_105:.*]], %[[VAL_106:.*]], %[[VAL_107:.*]] = ibis.pipeline.header
// CHECK:               %[[VAL_108:.*]], %[[VAL_109:.*]] = pipeline.unscheduled(%[[VAL_110:.*]] : i32 = %[[VAL_38]], %[[VAL_111:.*]] : i32 = %[[VAL_103]]) stall(%[[VAL_107]]) clock(%[[VAL_104]]) reset(%[[VAL_105]]) go(%[[VAL_106]]) entryEn(%[[VAL_112:.*]])  -> (out0 : i32) {
// CHECK:                 %[[VAL_113:.*]] = arith.subi %[[VAL_110]], %[[VAL_111]] : i32
// CHECK:                 pipeline.return %[[VAL_113]] : i32
// CHECK:               }
// CHECK:               ibis.sblock.return %[[VAL_114:.*]] : i32
// CHECK:             }
// CHECK:             %[[VAL_29]] = handshake.mux %[[VAL_115:.*]] {{\[}}%[[VAL_102]], %[[VAL_89]]] : index, i32
// CHECK:             %[[VAL_24]], %[[VAL_115]] = handshake.control_merge %[[VAL_86]], %[[VAL_85]] : none, index
// CHECK:             %[[VAL_26]] = ibis.sblock (%[[VAL_38]] : index = %[[VAL_51]], %[[VAL_116:.*]] : index = %[[VAL_35]]) -> index {
// CHECK:               %[[VAL_117:.*]], %[[VAL_118:.*]], %[[VAL_119:.*]], %[[VAL_120:.*]] = ibis.pipeline.header
// CHECK:               %[[VAL_121:.*]], %[[VAL_122:.*]] = pipeline.unscheduled(%[[VAL_123:.*]] : index = %[[VAL_38]], %[[VAL_124:.*]] : index = %[[VAL_116]]) stall(%[[VAL_120]]) clock(%[[VAL_117]]) reset(%[[VAL_118]]) go(%[[VAL_119]]) entryEn(%[[VAL_125:.*]])  -> (out0 : index) {
// CHECK:                 %[[VAL_126:.*]] = arith.addi %[[VAL_123]], %[[VAL_124]] : index
// CHECK:                 pipeline.return %[[VAL_126]] : index
// CHECK:               }
// CHECK:               ibis.sblock.return %[[VAL_127:.*]] : index
// CHECK:             }
// CHECK:             ibis.return %[[VAL_54]], %[[VAL_60]] : i32, none
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
