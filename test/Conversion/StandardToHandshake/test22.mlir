// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @if_for(
// CHECK-SAME:                           %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:4 = fork [4] %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1]]#1 {value = -1 : index} : index
// CHECK:           %[[VAL_4:.*]]:2 = fork [2] %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_4]]#0, %[[VAL_4]]#1 : index
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_1]]#0 {value = 20 : index} : index
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sge, %[[VAL_7]], %[[VAL_2]] : index
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = cond_br %[[VAL_8]], %[[VAL_1]]#3 : none
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = control_merge %[[VAL_9]] : none
// CHECK:           %[[VAL_13:.*]]:4 = fork [4] %[[VAL_11]] : none
// CHECK:           sink %[[VAL_12]] : index
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_13]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_15:.*]] = constant %[[VAL_13]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_16:.*]] = constant %[[VAL_13]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_13]]#3 : none
// CHECK:           %[[VAL_18:.*]] = br %[[VAL_14]] : index
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_15]] : index
// CHECK:           %[[VAL_20:.*]] = br %[[VAL_16]] : index
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = control_merge %[[VAL_17]] : none
// CHECK:           %[[VAL_23:.*]]:3 = fork [3] %[[VAL_22]] : index
// CHECK:           %[[VAL_24:.*]] = buffer [1] seq %[[VAL_25:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_26:.*]]:4 = fork [4] %[[VAL_24]] : i1
// CHECK:           %[[VAL_27:.*]] = mux %[[VAL_26]]#3 {{\[}}%[[VAL_21]], %[[VAL_28:.*]]] : i1, none
// CHECK:           %[[VAL_29:.*]] = mux %[[VAL_23]]#2 {{\[}}%[[VAL_19]]] : index, index
// CHECK:           %[[VAL_30:.*]] = mux %[[VAL_26]]#2 {{\[}}%[[VAL_29]], %[[VAL_31:.*]]] : i1, index
// CHECK:           %[[VAL_32:.*]]:2 = fork [2] %[[VAL_30]] : index
// CHECK:           %[[VAL_33:.*]] = mux %[[VAL_23]]#1 {{\[}}%[[VAL_20]]] : index, index
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_26]]#1 {{\[}}%[[VAL_33]], %[[VAL_35:.*]]] : i1, index
// CHECK:           %[[VAL_36:.*]] = mux %[[VAL_23]]#0 {{\[}}%[[VAL_18]]] : index, index
// CHECK:           %[[VAL_37:.*]] = mux %[[VAL_26]]#0 {{\[}}%[[VAL_36]], %[[VAL_38:.*]]] : i1, index
// CHECK:           %[[VAL_39:.*]]:2 = fork [2] %[[VAL_37]] : index
// CHECK:           %[[VAL_25]] = merge %[[VAL_40:.*]]#0 : i1
// CHECK:           %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_39]]#0, %[[VAL_32]]#0 : index
// CHECK:           %[[VAL_40]]:5 = fork [5] %[[VAL_41]] : i1
// CHECK:           %[[VAL_42:.*]], %[[VAL_43:.*]] = cond_br %[[VAL_40]]#4, %[[VAL_32]]#1 : index
// CHECK:           sink %[[VAL_43]] : index
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = cond_br %[[VAL_40]]#3, %[[VAL_34]] : index
// CHECK:           sink %[[VAL_45]] : index
// CHECK:           %[[VAL_46:.*]], %[[VAL_47:.*]] = cond_br %[[VAL_40]]#2, %[[VAL_27]] : none
// CHECK:           %[[VAL_48:.*]], %[[VAL_49:.*]] = cond_br %[[VAL_40]]#1, %[[VAL_39]]#1 : index
// CHECK:           sink %[[VAL_49]] : index
// CHECK:           %[[VAL_50:.*]] = merge %[[VAL_48]] : index
// CHECK:           %[[VAL_51:.*]]:2 = fork [2] %[[VAL_50]] : index
// CHECK:           %[[VAL_52:.*]] = merge %[[VAL_44]] : index
// CHECK:           %[[VAL_53:.*]] = merge %[[VAL_42]] : index
// CHECK:           %[[VAL_54:.*]], %[[VAL_55:.*]] = control_merge %[[VAL_46]] : none
// CHECK:           %[[VAL_56:.*]]:3 = fork [3] %[[VAL_54]] : none
// CHECK:           sink %[[VAL_55]] : index
// CHECK:           %[[VAL_57:.*]] = constant %[[VAL_56]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_58:.*]] = constant %[[VAL_56]]#0 {value = -10 : index} : index
// CHECK:           %[[VAL_59:.*]] = arith.addi %[[VAL_51]]#1, %[[VAL_58]] : index
// CHECK:           %[[VAL_60:.*]] = arith.cmpi sge, %[[VAL_59]], %[[VAL_57]] : index
// CHECK:           %[[VAL_61:.*]]:4 = fork [4] %[[VAL_60]] : i1
// CHECK:           %[[VAL_62:.*]], %[[VAL_63:.*]] = cond_br %[[VAL_61]]#3, %[[VAL_51]]#0 : index
// CHECK:           %[[VAL_64:.*]], %[[VAL_65:.*]] = cond_br %[[VAL_61]]#2, %[[VAL_52]] : index
// CHECK:           %[[VAL_66:.*]], %[[VAL_67:.*]] = cond_br %[[VAL_61]]#1, %[[VAL_53]] : index
// CHECK:           %[[VAL_68:.*]], %[[VAL_69:.*]] = cond_br %[[VAL_61]]#0, %[[VAL_56]]#2 : none
// CHECK:           %[[VAL_70:.*]] = merge %[[VAL_62]] : index
// CHECK:           %[[VAL_71:.*]] = merge %[[VAL_64]] : index
// CHECK:           %[[VAL_72:.*]] = merge %[[VAL_66]] : index
// CHECK:           %[[VAL_73:.*]], %[[VAL_74:.*]] = control_merge %[[VAL_68]] : none
// CHECK:           sink %[[VAL_74]] : index
// CHECK:           %[[VAL_75:.*]] = br %[[VAL_70]] : index
// CHECK:           %[[VAL_76:.*]] = br %[[VAL_71]] : index
// CHECK:           %[[VAL_77:.*]] = br %[[VAL_72]] : index
// CHECK:           %[[VAL_78:.*]] = br %[[VAL_73]] : none
// CHECK:           %[[VAL_79:.*]] = mux %[[VAL_80:.*]]#2 {{\[}}%[[VAL_75]], %[[VAL_63]]] : index, index
// CHECK:           %[[VAL_81:.*]] = mux %[[VAL_80]]#1 {{\[}}%[[VAL_76]], %[[VAL_65]]] : index, index
// CHECK:           %[[VAL_82:.*]]:2 = fork [2] %[[VAL_81]] : index
// CHECK:           %[[VAL_83:.*]] = mux %[[VAL_80]]#0 {{\[}}%[[VAL_77]], %[[VAL_67]]] : index, index
// CHECK:           %[[VAL_84:.*]], %[[VAL_85:.*]] = control_merge %[[VAL_78]], %[[VAL_69]] : none
// CHECK:           %[[VAL_80]]:3 = fork [3] %[[VAL_85]] : index
// CHECK:           %[[VAL_86:.*]] = arith.addi %[[VAL_79]], %[[VAL_82]]#1 : index
// CHECK:           %[[VAL_35]] = br %[[VAL_82]]#0 : index
// CHECK:           %[[VAL_31]] = br %[[VAL_83]] : index
// CHECK:           %[[VAL_28]] = br %[[VAL_84]] : none
// CHECK:           %[[VAL_38]] = br %[[VAL_86]] : index
// CHECK:           %[[VAL_87:.*]], %[[VAL_88:.*]] = control_merge %[[VAL_47]] : none
// CHECK:           sink %[[VAL_88]] : index
// CHECK:           %[[VAL_89:.*]] = br %[[VAL_87]] : none
// CHECK:           %[[VAL_90:.*]], %[[VAL_91:.*]] = control_merge %[[VAL_89]], %[[VAL_10]] : none
// CHECK:           %[[VAL_92:.*]]:4 = fork [4] %[[VAL_90]] : none
// CHECK:           sink %[[VAL_91]] : index
// CHECK:           %[[VAL_93:.*]] = constant %[[VAL_92]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_94:.*]] = constant %[[VAL_92]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_95:.*]] = constant %[[VAL_92]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_96:.*]] = br %[[VAL_92]]#3 : none
// CHECK:           %[[VAL_97:.*]] = br %[[VAL_93]] : index
// CHECK:           %[[VAL_98:.*]] = br %[[VAL_94]] : index
// CHECK:           %[[VAL_99:.*]] = br %[[VAL_95]] : index
// CHECK:           %[[VAL_100:.*]] = mux %[[VAL_101:.*]]#2 {{\[}}%[[VAL_102:.*]], %[[VAL_98]]] : index, index
// CHECK:           %[[VAL_103:.*]]:2 = fork [2] %[[VAL_100]] : index
// CHECK:           %[[VAL_104:.*]] = mux %[[VAL_101]]#1 {{\[}}%[[VAL_105:.*]], %[[VAL_99]]] : index, index
// CHECK:           %[[VAL_106:.*]], %[[VAL_107:.*]] = control_merge %[[VAL_108:.*]], %[[VAL_96]] : none
// CHECK:           %[[VAL_101]]:3 = fork [3] %[[VAL_107]] : index
// CHECK:           %[[VAL_109:.*]] = mux %[[VAL_101]]#0 {{\[}}%[[VAL_110:.*]], %[[VAL_97]]] : index, index
// CHECK:           %[[VAL_111:.*]]:2 = fork [2] %[[VAL_109]] : index
// CHECK:           %[[VAL_112:.*]] = arith.cmpi slt, %[[VAL_111]]#1, %[[VAL_103]]#1 : index
// CHECK:           %[[VAL_113:.*]]:4 = fork [4] %[[VAL_112]] : i1
// CHECK:           %[[VAL_114:.*]], %[[VAL_115:.*]] = cond_br %[[VAL_113]]#3, %[[VAL_103]]#0 : index
// CHECK:           sink %[[VAL_115]] : index
// CHECK:           %[[VAL_116:.*]], %[[VAL_117:.*]] = cond_br %[[VAL_113]]#2, %[[VAL_104]] : index
// CHECK:           sink %[[VAL_117]] : index
// CHECK:           %[[VAL_118:.*]], %[[VAL_119:.*]] = cond_br %[[VAL_113]]#1, %[[VAL_106]] : none
// CHECK:           %[[VAL_120:.*]], %[[VAL_121:.*]] = cond_br %[[VAL_113]]#0, %[[VAL_111]]#0 : index
// CHECK:           sink %[[VAL_121]] : index
// CHECK:           %[[VAL_122:.*]] = merge %[[VAL_120]] : index
// CHECK:           %[[VAL_123:.*]]:2 = fork [2] %[[VAL_122]] : index
// CHECK:           %[[VAL_124:.*]] = merge %[[VAL_116]] : index
// CHECK:           %[[VAL_125:.*]] = merge %[[VAL_114]] : index
// CHECK:           %[[VAL_126:.*]], %[[VAL_127:.*]] = control_merge %[[VAL_118]] : none
// CHECK:           %[[VAL_128:.*]]:3 = fork [3] %[[VAL_126]] : none
// CHECK:           sink %[[VAL_127]] : index
// CHECK:           %[[VAL_129:.*]] = constant %[[VAL_128]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_130:.*]] = constant %[[VAL_128]]#0 {value = -10 : index} : index
// CHECK:           %[[VAL_131:.*]] = arith.addi %[[VAL_123]]#1, %[[VAL_130]] : index
// CHECK:           %[[VAL_132:.*]] = arith.cmpi sge, %[[VAL_131]], %[[VAL_129]] : index
// CHECK:           %[[VAL_133:.*]]:4 = fork [4] %[[VAL_132]] : i1
// CHECK:           %[[VAL_134:.*]], %[[VAL_135:.*]] = cond_br %[[VAL_133]]#3, %[[VAL_123]]#0 : index
// CHECK:           %[[VAL_136:.*]], %[[VAL_137:.*]] = cond_br %[[VAL_133]]#2, %[[VAL_124]] : index
// CHECK:           %[[VAL_138:.*]], %[[VAL_139:.*]] = cond_br %[[VAL_133]]#1, %[[VAL_125]] : index
// CHECK:           %[[VAL_140:.*]], %[[VAL_141:.*]] = cond_br %[[VAL_133]]#0, %[[VAL_128]]#2 : none
// CHECK:           %[[VAL_142:.*]] = merge %[[VAL_134]] : index
// CHECK:           %[[VAL_143:.*]] = merge %[[VAL_136]] : index
// CHECK:           %[[VAL_144:.*]] = merge %[[VAL_138]] : index
// CHECK:           %[[VAL_145:.*]], %[[VAL_146:.*]] = control_merge %[[VAL_140]] : none
// CHECK:           %[[VAL_147:.*]]:4 = fork [4] %[[VAL_145]] : none
// CHECK:           sink %[[VAL_146]] : index
// CHECK:           %[[VAL_148:.*]] = constant %[[VAL_147]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_149:.*]] = constant %[[VAL_147]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_150:.*]] = constant %[[VAL_147]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_151:.*]] = br %[[VAL_142]] : index
// CHECK:           %[[VAL_152:.*]] = br %[[VAL_143]] : index
// CHECK:           %[[VAL_153:.*]] = br %[[VAL_144]] : index
// CHECK:           %[[VAL_154:.*]] = br %[[VAL_147]]#3 : none
// CHECK:           %[[VAL_155:.*]] = br %[[VAL_148]] : index
// CHECK:           %[[VAL_156:.*]] = br %[[VAL_149]] : index
// CHECK:           %[[VAL_157:.*]] = br %[[VAL_150]] : index
// CHECK:           %[[VAL_158:.*]] = mux %[[VAL_159:.*]]#5 {{\[}}%[[VAL_160:.*]], %[[VAL_156]]] : index, index
// CHECK:           %[[VAL_161:.*]]:2 = fork [2] %[[VAL_158]] : index
// CHECK:           %[[VAL_162:.*]] = mux %[[VAL_159]]#4 {{\[}}%[[VAL_163:.*]], %[[VAL_157]]] : index, index
// CHECK:           %[[VAL_164:.*]] = mux %[[VAL_159]]#3 {{\[}}%[[VAL_165:.*]], %[[VAL_151]]] : index, index
// CHECK:           %[[VAL_166:.*]] = mux %[[VAL_159]]#2 {{\[}}%[[VAL_167:.*]], %[[VAL_152]]] : index, index
// CHECK:           %[[VAL_168:.*]] = mux %[[VAL_159]]#1 {{\[}}%[[VAL_169:.*]], %[[VAL_153]]] : index, index
// CHECK:           %[[VAL_170:.*]], %[[VAL_171:.*]] = control_merge %[[VAL_172:.*]], %[[VAL_154]] : none
// CHECK:           %[[VAL_159]]:6 = fork [6] %[[VAL_171]] : index
// CHECK:           %[[VAL_173:.*]] = mux %[[VAL_159]]#0 {{\[}}%[[VAL_174:.*]], %[[VAL_155]]] : index, index
// CHECK:           %[[VAL_175:.*]]:2 = fork [2] %[[VAL_173]] : index
// CHECK:           %[[VAL_176:.*]] = arith.cmpi slt, %[[VAL_175]]#1, %[[VAL_161]]#1 : index
// CHECK:           %[[VAL_177:.*]]:7 = fork [7] %[[VAL_176]] : i1
// CHECK:           %[[VAL_178:.*]], %[[VAL_179:.*]] = cond_br %[[VAL_177]]#6, %[[VAL_161]]#0 : index
// CHECK:           sink %[[VAL_179]] : index
// CHECK:           %[[VAL_180:.*]], %[[VAL_181:.*]] = cond_br %[[VAL_177]]#5, %[[VAL_162]] : index
// CHECK:           sink %[[VAL_181]] : index
// CHECK:           %[[VAL_182:.*]], %[[VAL_183:.*]] = cond_br %[[VAL_177]]#4, %[[VAL_164]] : index
// CHECK:           %[[VAL_184:.*]], %[[VAL_185:.*]] = cond_br %[[VAL_177]]#3, %[[VAL_166]] : index
// CHECK:           %[[VAL_186:.*]], %[[VAL_187:.*]] = cond_br %[[VAL_177]]#2, %[[VAL_168]] : index
// CHECK:           %[[VAL_188:.*]], %[[VAL_189:.*]] = cond_br %[[VAL_177]]#1, %[[VAL_170]] : none
// CHECK:           %[[VAL_190:.*]], %[[VAL_191:.*]] = cond_br %[[VAL_177]]#0, %[[VAL_175]]#0 : index
// CHECK:           sink %[[VAL_191]] : index
// CHECK:           %[[VAL_192:.*]] = merge %[[VAL_190]] : index
// CHECK:           %[[VAL_193:.*]] = merge %[[VAL_180]] : index
// CHECK:           %[[VAL_194:.*]]:2 = fork [2] %[[VAL_193]] : index
// CHECK:           %[[VAL_195:.*]] = merge %[[VAL_178]] : index
// CHECK:           %[[VAL_196:.*]] = merge %[[VAL_182]] : index
// CHECK:           %[[VAL_197:.*]] = merge %[[VAL_184]] : index
// CHECK:           %[[VAL_198:.*]] = merge %[[VAL_186]] : index
// CHECK:           %[[VAL_199:.*]], %[[VAL_200:.*]] = control_merge %[[VAL_188]] : none
// CHECK:           sink %[[VAL_200]] : index
// CHECK:           %[[VAL_201:.*]] = arith.addi %[[VAL_192]], %[[VAL_194]]#1 : index
// CHECK:           %[[VAL_163]] = br %[[VAL_194]]#0 : index
// CHECK:           %[[VAL_160]] = br %[[VAL_195]] : index
// CHECK:           %[[VAL_165]] = br %[[VAL_196]] : index
// CHECK:           %[[VAL_167]] = br %[[VAL_197]] : index
// CHECK:           %[[VAL_169]] = br %[[VAL_198]] : index
// CHECK:           %[[VAL_172]] = br %[[VAL_199]] : none
// CHECK:           %[[VAL_174]] = br %[[VAL_201]] : index
// CHECK:           %[[VAL_202:.*]] = merge %[[VAL_183]] : index
// CHECK:           %[[VAL_203:.*]] = merge %[[VAL_185]] : index
// CHECK:           %[[VAL_204:.*]] = merge %[[VAL_187]] : index
// CHECK:           %[[VAL_205:.*]], %[[VAL_206:.*]] = control_merge %[[VAL_189]] : none
// CHECK:           sink %[[VAL_206]] : index
// CHECK:           %[[VAL_207:.*]] = br %[[VAL_202]] : index
// CHECK:           %[[VAL_208:.*]] = br %[[VAL_203]] : index
// CHECK:           %[[VAL_209:.*]] = br %[[VAL_204]] : index
// CHECK:           %[[VAL_210:.*]] = br %[[VAL_205]] : none
// CHECK:           %[[VAL_211:.*]] = mux %[[VAL_212:.*]]#2 {{\[}}%[[VAL_207]], %[[VAL_135]]] : index, index
// CHECK:           %[[VAL_213:.*]] = mux %[[VAL_212]]#1 {{\[}}%[[VAL_208]], %[[VAL_137]]] : index, index
// CHECK:           %[[VAL_214:.*]]:2 = fork [2] %[[VAL_213]] : index
// CHECK:           %[[VAL_215:.*]] = mux %[[VAL_212]]#0 {{\[}}%[[VAL_209]], %[[VAL_139]]] : index, index
// CHECK:           %[[VAL_216:.*]], %[[VAL_217:.*]] = control_merge %[[VAL_210]], %[[VAL_141]] : none
// CHECK:           %[[VAL_212]]:3 = fork [3] %[[VAL_217]] : index
// CHECK:           %[[VAL_218:.*]] = arith.addi %[[VAL_211]], %[[VAL_214]]#1 : index
// CHECK:           %[[VAL_105]] = br %[[VAL_214]]#0 : index
// CHECK:           %[[VAL_102]] = br %[[VAL_215]] : index
// CHECK:           %[[VAL_108]] = br %[[VAL_216]] : none
// CHECK:           %[[VAL_110]] = br %[[VAL_218]] : index
// CHECK:           %[[VAL_219:.*]], %[[VAL_220:.*]] = control_merge %[[VAL_119]] : none
// CHECK:           sink %[[VAL_220]] : index
// CHECK:           return %[[VAL_219]] : none
// CHECK:         }
func @if_for() {
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %1 = arith.muli %c-1, %c-1 : index
  %c20 = arith.constant 20 : index
  %2 = arith.addi %1, %c20 : index
  %3 = arith.cmpi sge, %2, %c0 : index
  cf.cond_br %3, ^bb1, ^bb7
^bb1: // pred: ^bb0
  %c0_0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb2(%c0_0 : index)
^bb2(%4: index):      // 2 preds: ^bb1, ^bb5
  %5 = arith.cmpi slt, %4, %c42 : index
  cf.cond_br %5, ^bb3, ^bb6
^bb3: // pred: ^bb2
  %c0_1 = arith.constant 0 : index
  %c-10 = arith.constant -10 : index
  %6 = arith.addi %4, %c-10 : index
  %7 = arith.cmpi sge, %6, %c0_1 : index
  cf.cond_br %7, ^bb4, ^bb5
^bb4: // pred: ^bb3
  cf.br ^bb5
^bb5: // 2 preds: ^bb3, ^bb4
  %8 = arith.addi %4, %c1 : index
  cf.br ^bb2(%8 : index)
^bb6: // pred: ^bb2
  cf.br ^bb7
^bb7: // 2 preds: ^bb0, ^bb6
  %c0_2 = arith.constant 0 : index
  %c42_3 = arith.constant 42 : index
  %c1_4 = arith.constant 1 : index
  cf.br ^bb8(%c0_2 : index)
^bb8(%9: index):      // 2 preds: ^bb7, ^bb14
  %10 = arith.cmpi slt, %9, %c42_3 : index
  cf.cond_br %10, ^bb9, ^bb15
^bb9: // pred: ^bb8
  %c0_5 = arith.constant 0 : index
  %c-10_6 = arith.constant -10 : index
  %11 = arith.addi %9, %c-10_6 : index
  %12 = arith.cmpi sge, %11, %c0_5 : index
  cf.cond_br %12, ^bb10, ^bb14
^bb10:        // pred: ^bb9
  %c0_7 = arith.constant 0 : index
  %c42_8 = arith.constant 42 : index
  %c1_9 = arith.constant 1 : index
  cf.br ^bb11(%c0_7 : index)
^bb11(%13: index):    // 2 preds: ^bb10, ^bb12
  %14 = arith.cmpi slt, %13, %c42_8 : index
  cf.cond_br %14, ^bb12, ^bb13
^bb12:        // pred: ^bb11
  %15 = arith.addi %13, %c1_9 : index
  cf.br ^bb11(%15 : index)
^bb13:        // pred: ^bb11
  cf.br ^bb14
^bb14:        // 2 preds: ^bb9, ^bb13
  %16 = arith.addi %9, %c1_4 : index
  cf.br ^bb8(%16 : index)
^bb15:        // pred: ^bb8
  return
  }
