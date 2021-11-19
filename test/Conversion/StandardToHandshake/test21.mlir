// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @loop_min_max(
// CHECK-SAME:                                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                                 %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_3:.*]]:4 = fork [4] %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_3]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_3]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_2]] : index
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_3]]#3 : none
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_4]] : index
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_6]] : index
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_13:.*]]#3 {{\[}}%[[VAL_14:.*]], %[[VAL_10]]] : index, index
// CHECK:           %[[VAL_15:.*]]:2 = fork [2] %[[VAL_12]] : index
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_13]]#2 {{\[}}%[[VAL_17:.*]], %[[VAL_7]]] : index, index
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_13]]#1 {{\[}}%[[VAL_19:.*]], %[[VAL_11]]] : index, index
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = control_merge %[[VAL_22:.*]], %[[VAL_8]] : none
// CHECK:           %[[VAL_13]]:4 = fork [4] %[[VAL_21]] : index
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_13]]#0 {{\[}}%[[VAL_24:.*]], %[[VAL_9]]] : index, index
// CHECK:           %[[VAL_25:.*]]:2 = fork [2] %[[VAL_23]] : index
// CHECK:           %[[VAL_26:.*]] = arith.cmpi slt, %[[VAL_25]]#1, %[[VAL_15]]#1 : index
// CHECK:           %[[VAL_27:.*]]:5 = fork [5] %[[VAL_26]] : i1
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_27]]#4, %[[VAL_15]]#0 : index
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_27]]#3, %[[VAL_16]] : index
// CHECK:           sink %[[VAL_31]] : index
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_27]]#2, %[[VAL_18]] : index
// CHECK:           sink %[[VAL_33]] : index
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = cond_br %[[VAL_27]]#1, %[[VAL_20]] : none
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = cond_br %[[VAL_27]]#0, %[[VAL_25]]#0 : index
// CHECK:           sink %[[VAL_37]] : index
// CHECK:           %[[VAL_38:.*]] = merge %[[VAL_36]] : index
// CHECK:           %[[VAL_39:.*]]:5 = fork [5] %[[VAL_38]] : index
// CHECK:           %[[VAL_40:.*]] = merge %[[VAL_30]] : index
// CHECK:           %[[VAL_41:.*]]:4 = fork [4] %[[VAL_40]] : index
// CHECK:           %[[VAL_42:.*]] = merge %[[VAL_32]] : index
// CHECK:           %[[VAL_43:.*]] = merge %[[VAL_28]] : index
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = control_merge %[[VAL_34]] : none
// CHECK:           %[[VAL_46:.*]]:4 = fork [4] %[[VAL_44]] : none
// CHECK:           sink %[[VAL_45]] : index
// CHECK:           %[[VAL_47:.*]] = constant %[[VAL_46]]#2 {value = -1 : index} : index
// CHECK:           %[[VAL_48:.*]] = arith.muli %[[VAL_39]]#4, %[[VAL_47]] : index
// CHECK:           %[[VAL_49:.*]] = arith.addi %[[VAL_48]], %[[VAL_41]]#3 : index
// CHECK:           %[[VAL_50:.*]]:2 = fork [2] %[[VAL_49]] : index
// CHECK:           %[[VAL_51:.*]] = arith.cmpi sgt, %[[VAL_39]]#3, %[[VAL_50]]#1 : index
// CHECK:           %[[VAL_52:.*]] = std.select %[[VAL_51]], %[[VAL_39]]#2, %[[VAL_50]]#0 : index
// CHECK:           %[[VAL_53:.*]] = constant %[[VAL_46]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_54:.*]] = arith.addi %[[VAL_39]]#1, %[[VAL_53]] : index
// CHECK:           %[[VAL_55:.*]]:2 = fork [2] %[[VAL_54]] : index
// CHECK:           %[[VAL_56:.*]] = arith.cmpi slt, %[[VAL_41]]#2, %[[VAL_55]]#1 : index
// CHECK:           %[[VAL_57:.*]] = std.select %[[VAL_56]], %[[VAL_41]]#1, %[[VAL_55]]#0 : index
// CHECK:           %[[VAL_58:.*]] = constant %[[VAL_46]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_59:.*]] = br %[[VAL_39]]#0 : index
// CHECK:           %[[VAL_60:.*]] = br %[[VAL_41]]#0 : index
// CHECK:           %[[VAL_61:.*]] = br %[[VAL_42]] : index
// CHECK:           %[[VAL_62:.*]] = br %[[VAL_43]] : index
// CHECK:           %[[VAL_63:.*]] = br %[[VAL_46]]#3 : none
// CHECK:           %[[VAL_64:.*]] = br %[[VAL_52]] : index
// CHECK:           %[[VAL_65:.*]] = br %[[VAL_57]] : index
// CHECK:           %[[VAL_66:.*]] = br %[[VAL_58]] : index
// CHECK:           %[[VAL_67:.*]] = mux %[[VAL_68:.*]]#6 {{\[}}%[[VAL_69:.*]], %[[VAL_65]]] : index, index
// CHECK:           %[[VAL_70:.*]]:2 = fork [2] %[[VAL_67]] : index
// CHECK:           %[[VAL_71:.*]] = mux %[[VAL_68]]#5 {{\[}}%[[VAL_72:.*]], %[[VAL_66]]] : index, index
// CHECK:           %[[VAL_73:.*]] = mux %[[VAL_68]]#4 {{\[}}%[[VAL_74:.*]], %[[VAL_59]]] : index, index
// CHECK:           %[[VAL_75:.*]] = mux %[[VAL_68]]#3 {{\[}}%[[VAL_76:.*]], %[[VAL_61]]] : index, index
// CHECK:           %[[VAL_77:.*]] = mux %[[VAL_68]]#2 {{\[}}%[[VAL_78:.*]], %[[VAL_62]]] : index, index
// CHECK:           %[[VAL_79:.*]] = mux %[[VAL_68]]#1 {{\[}}%[[VAL_80:.*]], %[[VAL_60]]] : index, index
// CHECK:           %[[VAL_81:.*]], %[[VAL_82:.*]] = control_merge %[[VAL_83:.*]], %[[VAL_63]] : none
// CHECK:           %[[VAL_68]]:7 = fork [7] %[[VAL_82]] : index
// CHECK:           %[[VAL_84:.*]] = mux %[[VAL_68]]#0 {{\[}}%[[VAL_85:.*]], %[[VAL_64]]] : index, index
// CHECK:           %[[VAL_86:.*]]:2 = fork [2] %[[VAL_84]] : index
// CHECK:           %[[VAL_87:.*]] = arith.cmpi slt, %[[VAL_86]]#1, %[[VAL_70]]#1 : index
// CHECK:           %[[VAL_88:.*]]:8 = fork [8] %[[VAL_87]] : i1
// CHECK:           %[[VAL_89:.*]], %[[VAL_90:.*]] = cond_br %[[VAL_88]]#7, %[[VAL_70]]#0 : index
// CHECK:           sink %[[VAL_90]] : index
// CHECK:           %[[VAL_91:.*]], %[[VAL_92:.*]] = cond_br %[[VAL_88]]#6, %[[VAL_71]] : index
// CHECK:           sink %[[VAL_92]] : index
// CHECK:           %[[VAL_93:.*]], %[[VAL_94:.*]] = cond_br %[[VAL_88]]#5, %[[VAL_73]] : index
// CHECK:           %[[VAL_95:.*]], %[[VAL_96:.*]] = cond_br %[[VAL_88]]#4, %[[VAL_75]] : index
// CHECK:           %[[VAL_97:.*]], %[[VAL_98:.*]] = cond_br %[[VAL_88]]#3, %[[VAL_77]] : index
// CHECK:           %[[VAL_99:.*]], %[[VAL_100:.*]] = cond_br %[[VAL_88]]#2, %[[VAL_79]] : index
// CHECK:           %[[VAL_101:.*]], %[[VAL_102:.*]] = cond_br %[[VAL_88]]#1, %[[VAL_81]] : none
// CHECK:           %[[VAL_103:.*]], %[[VAL_104:.*]] = cond_br %[[VAL_88]]#0, %[[VAL_86]]#0 : index
// CHECK:           sink %[[VAL_104]] : index
// CHECK:           %[[VAL_105:.*]] = merge %[[VAL_103]] : index
// CHECK:           %[[VAL_106:.*]] = merge %[[VAL_91]] : index
// CHECK:           %[[VAL_107:.*]]:2 = fork [2] %[[VAL_106]] : index
// CHECK:           %[[VAL_108:.*]] = merge %[[VAL_89]] : index
// CHECK:           %[[VAL_109:.*]] = merge %[[VAL_93]] : index
// CHECK:           %[[VAL_110:.*]] = merge %[[VAL_95]] : index
// CHECK:           %[[VAL_111:.*]] = merge %[[VAL_97]] : index
// CHECK:           %[[VAL_112:.*]] = merge %[[VAL_99]] : index
// CHECK:           %[[VAL_113:.*]], %[[VAL_114:.*]] = control_merge %[[VAL_101]] : none
// CHECK:           sink %[[VAL_114]] : index
// CHECK:           %[[VAL_115:.*]] = arith.addi %[[VAL_105]], %[[VAL_107]]#1 : index
// CHECK:           %[[VAL_72]] = br %[[VAL_107]]#0 : index
// CHECK:           %[[VAL_69]] = br %[[VAL_108]] : index
// CHECK:           %[[VAL_74]] = br %[[VAL_109]] : index
// CHECK:           %[[VAL_76]] = br %[[VAL_110]] : index
// CHECK:           %[[VAL_78]] = br %[[VAL_111]] : index
// CHECK:           %[[VAL_80]] = br %[[VAL_112]] : index
// CHECK:           %[[VAL_83]] = br %[[VAL_113]] : none
// CHECK:           %[[VAL_85]] = br %[[VAL_115]] : index
// CHECK:           %[[VAL_116:.*]] = merge %[[VAL_94]] : index
// CHECK:           %[[VAL_117:.*]] = merge %[[VAL_96]] : index
// CHECK:           %[[VAL_118:.*]]:2 = fork [2] %[[VAL_117]] : index
// CHECK:           %[[VAL_119:.*]] = merge %[[VAL_98]] : index
// CHECK:           %[[VAL_120:.*]] = merge %[[VAL_100]] : index
// CHECK:           %[[VAL_121:.*]], %[[VAL_122:.*]] = control_merge %[[VAL_102]] : none
// CHECK:           sink %[[VAL_122]] : index
// CHECK:           %[[VAL_123:.*]] = arith.addi %[[VAL_116]], %[[VAL_118]]#1 : index
// CHECK:           %[[VAL_19]] = br %[[VAL_118]]#0 : index
// CHECK:           %[[VAL_14]] = br %[[VAL_119]] : index
// CHECK:           %[[VAL_17]] = br %[[VAL_120]] : index
// CHECK:           %[[VAL_22]] = br %[[VAL_121]] : none
// CHECK:           %[[VAL_24]] = br %[[VAL_123]] : index
// CHECK:           %[[VAL_124:.*]], %[[VAL_125:.*]] = control_merge %[[VAL_35]] : none
// CHECK:           sink %[[VAL_125]] : index
// CHECK:           return %[[VAL_124]] : none
// CHECK:         }
func @loop_min_max(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : index
    %c1 = arith.constant 1 : index
    br ^bb1(%c0 : index)
  ^bb1(%0: index):      // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %c42 : index
    cond_br %1, ^bb2, ^bb6
  ^bb2: // pred: ^bb1
    %c-1 = arith.constant -1 : index
    %2 = arith.muli %0, %c-1 : index
    %3 = arith.addi %2, %arg0 : index
    %4 = arith.cmpi sgt, %0, %3 : index
    %5 = select %4, %0, %3 : index
    %c10 = arith.constant 10 : index
    %6 = arith.addi %0, %c10 : index
    %7 = arith.cmpi slt, %arg0, %6 : index
    %8 = select %7, %arg0, %6 : index
    %c1_0 = arith.constant 1 : index
    br ^bb3(%5 : index)
  ^bb3(%9: index):      // 2 preds: ^bb2, ^bb4
    %10 = arith.cmpi slt, %9, %8 : index
    cond_br %10, ^bb4, ^bb5
  ^bb4: // pred: ^bb3
    %11 = arith.addi %9, %c1_0 : index
    br ^bb3(%11 : index)
  ^bb5: // pred: ^bb3
    %12 = arith.addi %0, %c1 : index
    br ^bb1(%12 : index)
  ^bb6: // pred: ^bb1
    return
  }
