// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @more_imperfectly_nested_loops(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = control_merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]]:3 = fork [3] %[[VAL_2]] : none
// CHECK:           sink %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_4]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_4]]#0 {value = 42 : index} : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_4]]#2 : none
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_6]] : index
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_7]] : none
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = buffer [1] %[[VAL_14:.*]] {initValues = [0], sequential = true} : i1
// CHECK:           %[[VAL_15:.*]]:3 = fork [3] %[[VAL_13]] : i1
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_15]]#2 {{\[}}%[[VAL_10]], %[[VAL_17:.*]]] : i1, none
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_12]]#1 {{\[}}%[[VAL_9]]] : index, index
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_15]]#1 {{\[}}%[[VAL_18]], %[[VAL_20:.*]]] : i1, index
// CHECK:           %[[VAL_21:.*]]:2 = fork [2] %[[VAL_19]] : index
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_12]]#0 {{\[}}%[[VAL_8]]] : index, index
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_15]]#0 {{\[}}%[[VAL_22]], %[[VAL_24:.*]]] : i1, index
// CHECK:           %[[VAL_25:.*]]:2 = fork [2] %[[VAL_23]] : index
// CHECK:           %[[VAL_14]] = merge %[[VAL_26:.*]]#0 : i1
// CHECK:           %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_25]]#0, %[[VAL_21]]#0 : index
// CHECK:           %[[VAL_26]]:4 = fork [4] %[[VAL_27]] : i1
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_26]]#3, %[[VAL_21]]#1 : index
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_26]]#2, %[[VAL_16]] : none
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_26]]#1, %[[VAL_25]]#1 : index
// CHECK:           sink %[[VAL_33]] : index
// CHECK:           %[[VAL_34:.*]] = merge %[[VAL_32]] : index
// CHECK:           %[[VAL_35:.*]] = merge %[[VAL_28]] : index
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = control_merge %[[VAL_30]] : none
// CHECK:           sink %[[VAL_37]] : index
// CHECK:           %[[VAL_38:.*]] = br %[[VAL_34]] : index
// CHECK:           %[[VAL_39:.*]] = br %[[VAL_35]] : index
// CHECK:           %[[VAL_40:.*]] = br %[[VAL_36]] : none
// CHECK:           %[[VAL_41:.*]] = merge %[[VAL_38]] : index
// CHECK:           %[[VAL_42:.*]] = merge %[[VAL_39]] : index
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = control_merge %[[VAL_40]] : none
// CHECK:           %[[VAL_45:.*]]:3 = fork [3] %[[VAL_43]] : none
// CHECK:           sink %[[VAL_44]] : index
// CHECK:           %[[VAL_46:.*]] = constant %[[VAL_45]]#1 {value = 7 : index} : index
// CHECK:           %[[VAL_47:.*]] = constant %[[VAL_45]]#0 {value = 56 : index} : index
// CHECK:           %[[VAL_48:.*]] = br %[[VAL_41]] : index
// CHECK:           %[[VAL_49:.*]] = br %[[VAL_42]] : index
// CHECK:           %[[VAL_50:.*]] = br %[[VAL_45]]#2 : none
// CHECK:           %[[VAL_51:.*]] = br %[[VAL_46]] : index
// CHECK:           %[[VAL_52:.*]] = br %[[VAL_47]] : index
// CHECK:           %[[VAL_53:.*]] = mux %[[VAL_54:.*]]#3 {{\[}}%[[VAL_55:.*]], %[[VAL_52]]] : index, index
// CHECK:           %[[VAL_56:.*]]:2 = fork [2] %[[VAL_53]] : index
// CHECK:           %[[VAL_57:.*]] = mux %[[VAL_54]]#2 {{\[}}%[[VAL_58:.*]], %[[VAL_48]]] : index, index
// CHECK:           %[[VAL_59:.*]] = mux %[[VAL_54]]#1 {{\[}}%[[VAL_60:.*]], %[[VAL_49]]] : index, index
// CHECK:           %[[VAL_61:.*]], %[[VAL_62:.*]] = control_merge %[[VAL_63:.*]], %[[VAL_50]] : none
// CHECK:           %[[VAL_54]]:4 = fork [4] %[[VAL_62]] : index
// CHECK:           %[[VAL_64:.*]] = mux %[[VAL_54]]#0 {{\[}}%[[VAL_65:.*]], %[[VAL_51]]] : index, index
// CHECK:           %[[VAL_66:.*]]:2 = fork [2] %[[VAL_64]] : index
// CHECK:           %[[VAL_67:.*]] = arith.cmpi slt, %[[VAL_66]]#1, %[[VAL_56]]#1 : index
// CHECK:           %[[VAL_68:.*]]:5 = fork [5] %[[VAL_67]] : i1
// CHECK:           %[[VAL_69:.*]], %[[VAL_70:.*]] = cond_br %[[VAL_68]]#4, %[[VAL_56]]#0 : index
// CHECK:           sink %[[VAL_70]] : index
// CHECK:           %[[VAL_71:.*]], %[[VAL_72:.*]] = cond_br %[[VAL_68]]#3, %[[VAL_57]] : index
// CHECK:           %[[VAL_73:.*]], %[[VAL_74:.*]] = cond_br %[[VAL_68]]#2, %[[VAL_59]] : index
// CHECK:           %[[VAL_75:.*]], %[[VAL_76:.*]] = cond_br %[[VAL_68]]#1, %[[VAL_61]] : none
// CHECK:           %[[VAL_77:.*]], %[[VAL_78:.*]] = cond_br %[[VAL_68]]#0, %[[VAL_66]]#0 : index
// CHECK:           sink %[[VAL_78]] : index
// CHECK:           %[[VAL_79:.*]] = merge %[[VAL_77]] : index
// CHECK:           %[[VAL_80:.*]] = merge %[[VAL_69]] : index
// CHECK:           %[[VAL_81:.*]] = merge %[[VAL_71]] : index
// CHECK:           %[[VAL_82:.*]] = merge %[[VAL_73]] : index
// CHECK:           %[[VAL_83:.*]], %[[VAL_84:.*]] = control_merge %[[VAL_75]] : none
// CHECK:           %[[VAL_85:.*]]:2 = fork [2] %[[VAL_83]] : none
// CHECK:           sink %[[VAL_84]] : index
// CHECK:           %[[VAL_86:.*]] = constant %[[VAL_85]]#0 {value = 2 : index} : index
// CHECK:           %[[VAL_87:.*]] = arith.addi %[[VAL_79]], %[[VAL_86]] : index
// CHECK:           %[[VAL_55]] = br %[[VAL_80]] : index
// CHECK:           %[[VAL_58]] = br %[[VAL_81]] : index
// CHECK:           %[[VAL_60]] = br %[[VAL_82]] : index
// CHECK:           %[[VAL_63]] = br %[[VAL_85]]#1 : none
// CHECK:           %[[VAL_65]] = br %[[VAL_87]] : index
// CHECK:           %[[VAL_88:.*]] = merge %[[VAL_72]] : index
// CHECK:           %[[VAL_89:.*]] = merge %[[VAL_74]] : index
// CHECK:           %[[VAL_90:.*]], %[[VAL_91:.*]] = control_merge %[[VAL_76]] : none
// CHECK:           sink %[[VAL_91]] : index
// CHECK:           %[[VAL_92:.*]] = br %[[VAL_88]] : index
// CHECK:           %[[VAL_93:.*]] = br %[[VAL_89]] : index
// CHECK:           %[[VAL_94:.*]] = br %[[VAL_90]] : none
// CHECK:           %[[VAL_95:.*]] = merge %[[VAL_92]] : index
// CHECK:           %[[VAL_96:.*]] = merge %[[VAL_93]] : index
// CHECK:           %[[VAL_97:.*]], %[[VAL_98:.*]] = control_merge %[[VAL_94]] : none
// CHECK:           %[[VAL_99:.*]]:3 = fork [3] %[[VAL_97]] : none
// CHECK:           sink %[[VAL_98]] : index
// CHECK:           %[[VAL_100:.*]] = constant %[[VAL_99]]#1 {value = 18 : index} : index
// CHECK:           %[[VAL_101:.*]] = constant %[[VAL_99]]#0 {value = 37 : index} : index
// CHECK:           %[[VAL_102:.*]] = br %[[VAL_95]] : index
// CHECK:           %[[VAL_103:.*]] = br %[[VAL_96]] : index
// CHECK:           %[[VAL_104:.*]] = br %[[VAL_99]]#2 : none
// CHECK:           %[[VAL_105:.*]] = br %[[VAL_100]] : index
// CHECK:           %[[VAL_106:.*]] = br %[[VAL_101]] : index
// CHECK:           %[[VAL_107:.*]] = mux %[[VAL_108:.*]]#3 {{\[}}%[[VAL_109:.*]], %[[VAL_106]]] : index, index
// CHECK:           %[[VAL_110:.*]]:2 = fork [2] %[[VAL_107]] : index
// CHECK:           %[[VAL_111:.*]] = mux %[[VAL_108]]#2 {{\[}}%[[VAL_112:.*]], %[[VAL_102]]] : index, index
// CHECK:           %[[VAL_113:.*]] = mux %[[VAL_108]]#1 {{\[}}%[[VAL_114:.*]], %[[VAL_103]]] : index, index
// CHECK:           %[[VAL_115:.*]], %[[VAL_116:.*]] = control_merge %[[VAL_117:.*]], %[[VAL_104]] : none
// CHECK:           %[[VAL_108]]:4 = fork [4] %[[VAL_116]] : index
// CHECK:           %[[VAL_118:.*]] = mux %[[VAL_108]]#0 {{\[}}%[[VAL_119:.*]], %[[VAL_105]]] : index, index
// CHECK:           %[[VAL_120:.*]]:2 = fork [2] %[[VAL_118]] : index
// CHECK:           %[[VAL_121:.*]] = arith.cmpi slt, %[[VAL_120]]#1, %[[VAL_110]]#1 : index
// CHECK:           %[[VAL_122:.*]]:5 = fork [5] %[[VAL_121]] : i1
// CHECK:           %[[VAL_123:.*]], %[[VAL_124:.*]] = cond_br %[[VAL_122]]#4, %[[VAL_110]]#0 : index
// CHECK:           sink %[[VAL_124]] : index
// CHECK:           %[[VAL_125:.*]], %[[VAL_126:.*]] = cond_br %[[VAL_122]]#3, %[[VAL_111]] : index
// CHECK:           %[[VAL_127:.*]], %[[VAL_128:.*]] = cond_br %[[VAL_122]]#2, %[[VAL_113]] : index
// CHECK:           %[[VAL_129:.*]], %[[VAL_130:.*]] = cond_br %[[VAL_122]]#1, %[[VAL_115]] : none
// CHECK:           %[[VAL_131:.*]], %[[VAL_132:.*]] = cond_br %[[VAL_122]]#0, %[[VAL_120]]#0 : index
// CHECK:           sink %[[VAL_132]] : index
// CHECK:           %[[VAL_133:.*]] = merge %[[VAL_131]] : index
// CHECK:           %[[VAL_134:.*]] = merge %[[VAL_123]] : index
// CHECK:           %[[VAL_135:.*]] = merge %[[VAL_125]] : index
// CHECK:           %[[VAL_136:.*]] = merge %[[VAL_127]] : index
// CHECK:           %[[VAL_137:.*]], %[[VAL_138:.*]] = control_merge %[[VAL_129]] : none
// CHECK:           %[[VAL_139:.*]]:2 = fork [2] %[[VAL_137]] : none
// CHECK:           sink %[[VAL_138]] : index
// CHECK:           %[[VAL_140:.*]] = constant %[[VAL_139]]#0 {value = 3 : index} : index
// CHECK:           %[[VAL_141:.*]] = arith.addi %[[VAL_133]], %[[VAL_140]] : index
// CHECK:           %[[VAL_109]] = br %[[VAL_134]] : index
// CHECK:           %[[VAL_112]] = br %[[VAL_135]] : index
// CHECK:           %[[VAL_114]] = br %[[VAL_136]] : index
// CHECK:           %[[VAL_117]] = br %[[VAL_139]]#1 : none
// CHECK:           %[[VAL_119]] = br %[[VAL_141]] : index
// CHECK:           %[[VAL_142:.*]] = merge %[[VAL_126]] : index
// CHECK:           %[[VAL_143:.*]] = merge %[[VAL_128]] : index
// CHECK:           %[[VAL_144:.*]], %[[VAL_145:.*]] = control_merge %[[VAL_130]] : none
// CHECK:           %[[VAL_146:.*]]:2 = fork [2] %[[VAL_144]] : none
// CHECK:           sink %[[VAL_145]] : index
// CHECK:           %[[VAL_147:.*]] = constant %[[VAL_146]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_148:.*]] = arith.addi %[[VAL_142]], %[[VAL_147]] : index
// CHECK:           %[[VAL_20]] = br %[[VAL_143]] : index
// CHECK:           %[[VAL_17]] = br %[[VAL_146]]#1 : none
// CHECK:           %[[VAL_24]] = br %[[VAL_148]] : index
// CHECK:           %[[VAL_149:.*]], %[[VAL_150:.*]] = control_merge %[[VAL_31]] : none
// CHECK:           sink %[[VAL_150]] : index
// CHECK:           return %[[VAL_149]] : none
// CHECK:         }
func @more_imperfectly_nested_loops() {
^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  br ^bb2(%c0 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb11
  %1 = arith.cmpi slt, %0, %c42 : index
  cond_br %1, ^bb3, ^bb12
^bb3:	// pred: ^bb2
  br ^bb4
^bb4:	// pred: ^bb3
  %c7 = arith.constant 7 : index
  %c56 = arith.constant 56 : index
  br ^bb5(%c7 : index)
^bb5(%2: index):	// 2 preds: ^bb4, ^bb6
  %3 = arith.cmpi slt, %2, %c56 : index
  cond_br %3, ^bb6, ^bb7
^bb6:	// pred: ^bb5
  %c2 = arith.constant 2 : index
  %4 = arith.addi %2, %c2 : index
  br ^bb5(%4 : index)
^bb7:	// pred: ^bb5
  br ^bb8
^bb8:	// pred: ^bb7
  %c18 = arith.constant 18 : index
  %c37 = arith.constant 37 : index
  br ^bb9(%c18 : index)
^bb9(%5: index):	// 2 preds: ^bb8, ^bb10
  %6 = arith.cmpi slt, %5, %c37 : index
  cond_br %6, ^bb10, ^bb11
^bb10:	// pred: ^bb9
  %c3 = arith.constant 3 : index
  %7 = arith.addi %5, %c3 : index
  br ^bb9(%7 : index)
^bb11:	// pred: ^bb9
  %c1 = arith.constant 1 : index
  %8 = arith.addi %0, %c1 : index
  br ^bb2(%8 : index)
^bb12:	// pred: ^bb2
  return
}
