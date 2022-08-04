// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

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
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]]#0 : i1 to index
// CHECK:           %[[VAL_11:.*]] = buffer [2] fifo %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]] = br %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_9]]#1, %[[VAL_1]]#3 : none
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_13]] : none
// CHECK:           %[[VAL_17:.*]]:4 = fork [4] %[[VAL_15]] : none
// CHECK:           sink %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]] = constant %[[VAL_17]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_19:.*]] = constant %[[VAL_17]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_20:.*]] = constant %[[VAL_17]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_21:.*]] = br %[[VAL_17]]#3 : none
// CHECK:           %[[VAL_22:.*]] = br %[[VAL_18]] : index
// CHECK:           %[[VAL_23:.*]] = br %[[VAL_19]] : index
// CHECK:           %[[VAL_24:.*]] = br %[[VAL_20]] : index
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = control_merge %[[VAL_21]] : none
// CHECK:           %[[VAL_27:.*]]:3 = fork [3] %[[VAL_26]] : index
// CHECK:           %[[VAL_28:.*]] = buffer [1] seq %[[VAL_29:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_30:.*]]:4 = fork [4] %[[VAL_28]] : i1
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_30]]#3 {{\[}}%[[VAL_25]], %[[VAL_32:.*]]] : i1, none
// CHECK:           %[[VAL_33:.*]] = mux %[[VAL_27]]#2 {{\[}}%[[VAL_23]]] : index, index
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_30]]#2 {{\[}}%[[VAL_33]], %[[VAL_35:.*]]] : i1, index
// CHECK:           %[[VAL_36:.*]]:2 = fork [2] %[[VAL_34]] : index
// CHECK:           %[[VAL_37:.*]] = mux %[[VAL_27]]#1 {{\[}}%[[VAL_24]]] : index, index
// CHECK:           %[[VAL_38:.*]] = mux %[[VAL_30]]#1 {{\[}}%[[VAL_37]], %[[VAL_39:.*]]] : i1, index
// CHECK:           %[[VAL_40:.*]] = mux %[[VAL_27]]#0 {{\[}}%[[VAL_22]]] : index, index
// CHECK:           %[[VAL_41:.*]] = mux %[[VAL_30]]#0 {{\[}}%[[VAL_40]], %[[VAL_42:.*]]] : i1, index
// CHECK:           %[[VAL_43:.*]]:2 = fork [2] %[[VAL_41]] : index
// CHECK:           %[[VAL_29]] = merge %[[VAL_44:.*]]#0 : i1
// CHECK:           %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_43]]#0, %[[VAL_36]]#0 : index
// CHECK:           %[[VAL_44]]:5 = fork [5] %[[VAL_45]] : i1
// CHECK:           %[[VAL_46:.*]], %[[VAL_47:.*]] = cond_br %[[VAL_44]]#4, %[[VAL_36]]#1 : index
// CHECK:           sink %[[VAL_47]] : index
// CHECK:           %[[VAL_48:.*]], %[[VAL_49:.*]] = cond_br %[[VAL_44]]#3, %[[VAL_38]] : index
// CHECK:           sink %[[VAL_49]] : index
// CHECK:           %[[VAL_50:.*]], %[[VAL_51:.*]] = cond_br %[[VAL_44]]#2, %[[VAL_31]] : none
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = cond_br %[[VAL_44]]#1, %[[VAL_43]]#1 : index
// CHECK:           sink %[[VAL_53]] : index
// CHECK:           %[[VAL_54:.*]] = merge %[[VAL_52]] : index
// CHECK:           %[[VAL_55:.*]]:2 = fork [2] %[[VAL_54]] : index
// CHECK:           %[[VAL_56:.*]] = merge %[[VAL_48]] : index
// CHECK:           %[[VAL_57:.*]] = merge %[[VAL_46]] : index
// CHECK:           %[[VAL_58:.*]], %[[VAL_59:.*]] = control_merge %[[VAL_50]] : none
// CHECK:           %[[VAL_60:.*]]:3 = fork [3] %[[VAL_58]] : none
// CHECK:           sink %[[VAL_59]] : index
// CHECK:           %[[VAL_61:.*]] = constant %[[VAL_60]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_62:.*]] = constant %[[VAL_60]]#0 {value = -10 : index} : index
// CHECK:           %[[VAL_63:.*]] = arith.addi %[[VAL_55]]#1, %[[VAL_62]] : index
// CHECK:           %[[VAL_64:.*]] = arith.cmpi sge, %[[VAL_63]], %[[VAL_61]] : index
// CHECK:           %[[VAL_65:.*]]:4 = fork [4] %[[VAL_64]] : i1
// CHECK:           %[[VAL_66:.*]], %[[VAL_67:.*]] = cond_br %[[VAL_65]]#3, %[[VAL_55]]#0 : index
// CHECK:           %[[VAL_68:.*]], %[[VAL_69:.*]] = cond_br %[[VAL_65]]#2, %[[VAL_56]] : index
// CHECK:           %[[VAL_70:.*]], %[[VAL_71:.*]] = cond_br %[[VAL_65]]#1, %[[VAL_57]] : index
// CHECK:           %[[VAL_72:.*]], %[[VAL_73:.*]] = cond_br %[[VAL_65]]#0, %[[VAL_60]]#2 : none
// CHECK:           %[[VAL_74:.*]] = merge %[[VAL_66]] : index
// CHECK:           %[[VAL_75:.*]] = merge %[[VAL_68]] : index
// CHECK:           %[[VAL_76:.*]] = merge %[[VAL_70]] : index
// CHECK:           %[[VAL_77:.*]], %[[VAL_78:.*]] = control_merge %[[VAL_72]] : none
// CHECK:           sink %[[VAL_78]] : index
// CHECK:           %[[VAL_79:.*]] = br %[[VAL_74]] : index
// CHECK:           %[[VAL_80:.*]] = br %[[VAL_75]] : index
// CHECK:           %[[VAL_81:.*]] = br %[[VAL_76]] : index
// CHECK:           %[[VAL_82:.*]] = br %[[VAL_77]] : none
// CHECK:           %[[VAL_83:.*]] = mux %[[VAL_84:.*]]#2 {{\[}}%[[VAL_79]], %[[VAL_67]]] : index, index
// CHECK:           %[[VAL_85:.*]] = mux %[[VAL_84]]#1 {{\[}}%[[VAL_80]], %[[VAL_69]]] : index, index
// CHECK:           %[[VAL_86:.*]]:2 = fork [2] %[[VAL_85]] : index
// CHECK:           %[[VAL_87:.*]] = mux %[[VAL_84]]#0 {{\[}}%[[VAL_81]], %[[VAL_71]]] : index, index
// CHECK:           %[[VAL_88:.*]], %[[VAL_89:.*]] = control_merge %[[VAL_82]], %[[VAL_73]] : none
// CHECK:           %[[VAL_84]]:3 = fork [3] %[[VAL_89]] : index
// CHECK:           %[[VAL_90:.*]] = arith.addi %[[VAL_83]], %[[VAL_86]]#1 : index
// CHECK:           %[[VAL_39]] = br %[[VAL_86]]#0 : index
// CHECK:           %[[VAL_35]] = br %[[VAL_87]] : index
// CHECK:           %[[VAL_32]] = br %[[VAL_88]] : none
// CHECK:           %[[VAL_42]] = br %[[VAL_90]] : index
// CHECK:           %[[VAL_91:.*]], %[[VAL_92:.*]] = control_merge %[[VAL_51]] : none
// CHECK:           sink %[[VAL_92]] : index
// CHECK:           %[[VAL_93:.*]] = br %[[VAL_91]] : none
// CHECK:           %[[VAL_94:.*]] = mux %[[VAL_12]] {{\[}}%[[VAL_93]], %[[VAL_14]]] : index, none
// CHECK:           %[[VAL_95:.*]]:4 = fork [4] %[[VAL_94]] : none
// CHECK:           %[[VAL_96:.*]] = constant %[[VAL_95]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_97:.*]] = constant %[[VAL_95]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_98:.*]] = constant %[[VAL_95]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_99:.*]] = br %[[VAL_95]]#3 : none
// CHECK:           %[[VAL_100:.*]] = br %[[VAL_96]] : index
// CHECK:           %[[VAL_101:.*]] = br %[[VAL_97]] : index
// CHECK:           %[[VAL_102:.*]] = br %[[VAL_98]] : index
// CHECK:           %[[VAL_103:.*]], %[[VAL_104:.*]] = control_merge %[[VAL_99]] : none
// CHECK:           %[[VAL_105:.*]]:3 = fork [3] %[[VAL_104]] : index
// CHECK:           %[[VAL_106:.*]] = buffer [1] seq %[[VAL_107:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_108:.*]]:4 = fork [4] %[[VAL_106]] : i1
// CHECK:           %[[VAL_109:.*]] = mux %[[VAL_108]]#3 {{\[}}%[[VAL_103]], %[[VAL_110:.*]]] : i1, none
// CHECK:           %[[VAL_111:.*]] = mux %[[VAL_105]]#2 {{\[}}%[[VAL_101]]] : index, index
// CHECK:           %[[VAL_112:.*]] = mux %[[VAL_108]]#2 {{\[}}%[[VAL_111]], %[[VAL_113:.*]]] : i1, index
// CHECK:           %[[VAL_114:.*]]:2 = fork [2] %[[VAL_112]] : index
// CHECK:           %[[VAL_115:.*]] = mux %[[VAL_105]]#1 {{\[}}%[[VAL_102]]] : index, index
// CHECK:           %[[VAL_116:.*]] = mux %[[VAL_108]]#1 {{\[}}%[[VAL_115]], %[[VAL_117:.*]]] : i1, index
// CHECK:           %[[VAL_118:.*]] = mux %[[VAL_105]]#0 {{\[}}%[[VAL_100]]] : index, index
// CHECK:           %[[VAL_119:.*]] = mux %[[VAL_108]]#0 {{\[}}%[[VAL_118]], %[[VAL_120:.*]]] : i1, index
// CHECK:           %[[VAL_121:.*]]:2 = fork [2] %[[VAL_119]] : index
// CHECK:           %[[VAL_107]] = merge %[[VAL_122:.*]]#0 : i1
// CHECK:           %[[VAL_123:.*]] = arith.cmpi slt, %[[VAL_121]]#0, %[[VAL_114]]#0 : index
// CHECK:           %[[VAL_122]]:5 = fork [5] %[[VAL_123]] : i1
// CHECK:           %[[VAL_124:.*]], %[[VAL_125:.*]] = cond_br %[[VAL_122]]#4, %[[VAL_114]]#1 : index
// CHECK:           sink %[[VAL_125]] : index
// CHECK:           %[[VAL_126:.*]], %[[VAL_127:.*]] = cond_br %[[VAL_122]]#3, %[[VAL_116]] : index
// CHECK:           sink %[[VAL_127]] : index
// CHECK:           %[[VAL_128:.*]], %[[VAL_129:.*]] = cond_br %[[VAL_122]]#2, %[[VAL_109]] : none
// CHECK:           %[[VAL_130:.*]], %[[VAL_131:.*]] = cond_br %[[VAL_122]]#1, %[[VAL_121]]#1 : index
// CHECK:           sink %[[VAL_131]] : index
// CHECK:           %[[VAL_132:.*]] = merge %[[VAL_130]] : index
// CHECK:           %[[VAL_133:.*]]:2 = fork [2] %[[VAL_132]] : index
// CHECK:           %[[VAL_134:.*]] = merge %[[VAL_126]] : index
// CHECK:           %[[VAL_135:.*]] = merge %[[VAL_124]] : index
// CHECK:           %[[VAL_136:.*]], %[[VAL_137:.*]] = control_merge %[[VAL_128]] : none
// CHECK:           %[[VAL_138:.*]]:3 = fork [3] %[[VAL_136]] : none
// CHECK:           sink %[[VAL_137]] : index
// CHECK:           %[[VAL_139:.*]] = constant %[[VAL_138]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_140:.*]] = constant %[[VAL_138]]#0 {value = -10 : index} : index
// CHECK:           %[[VAL_141:.*]] = arith.addi %[[VAL_133]]#1, %[[VAL_140]] : index
// CHECK:           %[[VAL_142:.*]] = arith.cmpi sge, %[[VAL_141]], %[[VAL_139]] : index
// CHECK:           %[[VAL_143:.*]]:4 = fork [4] %[[VAL_142]] : i1
// CHECK:           %[[VAL_144:.*]], %[[VAL_145:.*]] = cond_br %[[VAL_143]]#3, %[[VAL_133]]#0 : index
// CHECK:           %[[VAL_146:.*]], %[[VAL_147:.*]] = cond_br %[[VAL_143]]#2, %[[VAL_134]] : index
// CHECK:           %[[VAL_148:.*]], %[[VAL_149:.*]] = cond_br %[[VAL_143]]#1, %[[VAL_135]] : index
// CHECK:           %[[VAL_150:.*]], %[[VAL_151:.*]] = cond_br %[[VAL_143]]#0, %[[VAL_138]]#2 : none
// CHECK:           %[[VAL_152:.*]] = merge %[[VAL_144]] : index
// CHECK:           %[[VAL_153:.*]] = merge %[[VAL_146]] : index
// CHECK:           %[[VAL_154:.*]] = merge %[[VAL_148]] : index
// CHECK:           %[[VAL_155:.*]], %[[VAL_156:.*]] = control_merge %[[VAL_150]] : none
// CHECK:           %[[VAL_157:.*]]:4 = fork [4] %[[VAL_155]] : none
// CHECK:           sink %[[VAL_156]] : index
// CHECK:           %[[VAL_158:.*]] = constant %[[VAL_157]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_159:.*]] = constant %[[VAL_157]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_160:.*]] = constant %[[VAL_157]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_161:.*]] = br %[[VAL_152]] : index
// CHECK:           %[[VAL_162:.*]] = br %[[VAL_153]] : index
// CHECK:           %[[VAL_163:.*]] = br %[[VAL_154]] : index
// CHECK:           %[[VAL_164:.*]] = br %[[VAL_157]]#3 : none
// CHECK:           %[[VAL_165:.*]] = br %[[VAL_158]] : index
// CHECK:           %[[VAL_166:.*]] = br %[[VAL_159]] : index
// CHECK:           %[[VAL_167:.*]] = br %[[VAL_160]] : index
// CHECK:           %[[VAL_168:.*]] = mux %[[VAL_169:.*]]#5 {{\[}}%[[VAL_170:.*]], %[[VAL_166]]] : index, index
// CHECK:           %[[VAL_171:.*]]:2 = fork [2] %[[VAL_168]] : index
// CHECK:           %[[VAL_172:.*]] = mux %[[VAL_169]]#4 {{\[}}%[[VAL_173:.*]], %[[VAL_167]]] : index, index
// CHECK:           %[[VAL_174:.*]] = mux %[[VAL_169]]#3 {{\[}}%[[VAL_175:.*]], %[[VAL_161]]] : index, index
// CHECK:           %[[VAL_176:.*]] = mux %[[VAL_169]]#2 {{\[}}%[[VAL_177:.*]], %[[VAL_162]]] : index, index
// CHECK:           %[[VAL_178:.*]] = mux %[[VAL_169]]#1 {{\[}}%[[VAL_179:.*]], %[[VAL_163]]] : index, index
// CHECK:           %[[VAL_180:.*]], %[[VAL_181:.*]] = control_merge %[[VAL_182:.*]], %[[VAL_164]] : none
// CHECK:           %[[VAL_169]]:6 = fork [6] %[[VAL_181]] : index
// CHECK:           %[[VAL_183:.*]] = mux %[[VAL_169]]#0 {{\[}}%[[VAL_184:.*]], %[[VAL_165]]] : index, index
// CHECK:           %[[VAL_185:.*]]:2 = fork [2] %[[VAL_183]] : index
// CHECK:           %[[VAL_186:.*]] = arith.cmpi slt, %[[VAL_185]]#1, %[[VAL_171]]#1 : index
// CHECK:           %[[VAL_187:.*]]:7 = fork [7] %[[VAL_186]] : i1
// CHECK:           %[[VAL_188:.*]], %[[VAL_189:.*]] = cond_br %[[VAL_187]]#6, %[[VAL_171]]#0 : index
// CHECK:           sink %[[VAL_189]] : index
// CHECK:           %[[VAL_190:.*]], %[[VAL_191:.*]] = cond_br %[[VAL_187]]#5, %[[VAL_172]] : index
// CHECK:           sink %[[VAL_191]] : index
// CHECK:           %[[VAL_192:.*]], %[[VAL_193:.*]] = cond_br %[[VAL_187]]#4, %[[VAL_174]] : index
// CHECK:           %[[VAL_194:.*]], %[[VAL_195:.*]] = cond_br %[[VAL_187]]#3, %[[VAL_176]] : index
// CHECK:           %[[VAL_196:.*]], %[[VAL_197:.*]] = cond_br %[[VAL_187]]#2, %[[VAL_178]] : index
// CHECK:           %[[VAL_198:.*]], %[[VAL_199:.*]] = cond_br %[[VAL_187]]#1, %[[VAL_180]] : none
// CHECK:           %[[VAL_200:.*]], %[[VAL_201:.*]] = cond_br %[[VAL_187]]#0, %[[VAL_185]]#0 : index
// CHECK:           sink %[[VAL_201]] : index
// CHECK:           %[[VAL_202:.*]] = merge %[[VAL_200]] : index
// CHECK:           %[[VAL_203:.*]] = merge %[[VAL_190]] : index
// CHECK:           %[[VAL_204:.*]]:2 = fork [2] %[[VAL_203]] : index
// CHECK:           %[[VAL_205:.*]] = merge %[[VAL_188]] : index
// CHECK:           %[[VAL_206:.*]] = merge %[[VAL_192]] : index
// CHECK:           %[[VAL_207:.*]] = merge %[[VAL_194]] : index
// CHECK:           %[[VAL_208:.*]] = merge %[[VAL_196]] : index
// CHECK:           %[[VAL_209:.*]], %[[VAL_210:.*]] = control_merge %[[VAL_198]] : none
// CHECK:           sink %[[VAL_210]] : index
// CHECK:           %[[VAL_211:.*]] = arith.addi %[[VAL_202]], %[[VAL_204]]#1 : index
// CHECK:           %[[VAL_173]] = br %[[VAL_204]]#0 : index
// CHECK:           %[[VAL_170]] = br %[[VAL_205]] : index
// CHECK:           %[[VAL_175]] = br %[[VAL_206]] : index
// CHECK:           %[[VAL_177]] = br %[[VAL_207]] : index
// CHECK:           %[[VAL_179]] = br %[[VAL_208]] : index
// CHECK:           %[[VAL_182]] = br %[[VAL_209]] : none
// CHECK:           %[[VAL_184]] = br %[[VAL_211]] : index
// CHECK:           %[[VAL_212:.*]] = merge %[[VAL_193]] : index
// CHECK:           %[[VAL_213:.*]] = merge %[[VAL_195]] : index
// CHECK:           %[[VAL_214:.*]] = merge %[[VAL_197]] : index
// CHECK:           %[[VAL_215:.*]], %[[VAL_216:.*]] = control_merge %[[VAL_199]] : none
// CHECK:           sink %[[VAL_216]] : index
// CHECK:           %[[VAL_217:.*]] = br %[[VAL_212]] : index
// CHECK:           %[[VAL_218:.*]] = br %[[VAL_213]] : index
// CHECK:           %[[VAL_219:.*]] = br %[[VAL_214]] : index
// CHECK:           %[[VAL_220:.*]] = br %[[VAL_215]] : none
// CHECK:           %[[VAL_221:.*]] = mux %[[VAL_222:.*]]#2 {{\[}}%[[VAL_217]], %[[VAL_145]]] : index, index
// CHECK:           %[[VAL_223:.*]] = mux %[[VAL_222]]#1 {{\[}}%[[VAL_218]], %[[VAL_147]]] : index, index
// CHECK:           %[[VAL_224:.*]]:2 = fork [2] %[[VAL_223]] : index
// CHECK:           %[[VAL_225:.*]] = mux %[[VAL_222]]#0 {{\[}}%[[VAL_219]], %[[VAL_149]]] : index, index
// CHECK:           %[[VAL_226:.*]], %[[VAL_227:.*]] = control_merge %[[VAL_220]], %[[VAL_151]] : none
// CHECK:           %[[VAL_222]]:3 = fork [3] %[[VAL_227]] : index
// CHECK:           %[[VAL_228:.*]] = arith.addi %[[VAL_221]], %[[VAL_224]]#1 : index
// CHECK:           %[[VAL_117]] = br %[[VAL_224]]#0 : index
// CHECK:           %[[VAL_113]] = br %[[VAL_225]] : index
// CHECK:           %[[VAL_110]] = br %[[VAL_226]] : none
// CHECK:           %[[VAL_120]] = br %[[VAL_228]] : index
// CHECK:           %[[VAL_229:.*]], %[[VAL_230:.*]] = control_merge %[[VAL_129]] : none
// CHECK:           sink %[[VAL_230]] : index
// CHECK:           return %[[VAL_229]] : none
// CHECK:         }
func.func @if_for() {
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
