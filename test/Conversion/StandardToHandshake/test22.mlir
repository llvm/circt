// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
  func @if_for() {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @if_for(
// CHECK-SAME:                           %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:4 = "handshake.fork"(%[[VAL_0]]) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_2:.*]] = "handshake.constant"(%[[VAL_1]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_3:.*]] = "handshake.constant"(%[[VAL_1]]#1) {value = -1 : index} : (none) -> index
// CHECK:           %[[VAL_4:.*]]:2 = "handshake.fork"(%[[VAL_3]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_4]]#0, %[[VAL_4]]#1 : index
// CHECK:           %[[VAL_6:.*]] = "handshake.constant"(%[[VAL_1]]#0) {value = 20 : index} : (none) -> index
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sge, %[[VAL_7]], %[[VAL_2]] : index
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = "handshake.conditional_branch"(%[[VAL_8]], %[[VAL_1]]#3) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_11:.*]]:2 = "handshake.control_merge"(%[[VAL_9]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_12:.*]]:4 = "handshake.fork"(%[[VAL_11]]#0) {control = true} : (none) -> (none, none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_11]]#1) : (index) -> ()
// CHECK:           %[[VAL_13:.*]] = "handshake.constant"(%[[VAL_12]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_14:.*]] = "handshake.constant"(%[[VAL_12]]#1) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_15:.*]] = "handshake.constant"(%[[VAL_12]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_16:.*]] = "handshake.branch"(%[[VAL_12]]#3) {control = true} : (none) -> none
// CHECK:           %[[VAL_17:.*]] = "handshake.branch"(%[[VAL_13]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_18:.*]] = "handshake.branch"(%[[VAL_14]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_19:.*]] = "handshake.branch"(%[[VAL_15]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_20:.*]] = "handshake.mux"(%[[VAL_21:.*]]#2, %[[VAL_22:.*]], %[[VAL_18]]) : (index, index, index) -> index
// CHECK:           %[[VAL_23:.*]]:2 = "handshake.fork"(%[[VAL_20]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_24:.*]] = "handshake.mux"(%[[VAL_21]]#1, %[[VAL_25:.*]], %[[VAL_19]]) : (index, index, index) -> index
// CHECK:           %[[VAL_26:.*]]:2 = "handshake.control_merge"(%[[VAL_27:.*]], %[[VAL_16]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_21]]:3 = "handshake.fork"(%[[VAL_26]]#1) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_28:.*]] = "handshake.mux"(%[[VAL_21]]#0, %[[VAL_29:.*]], %[[VAL_17]]) : (index, index, index) -> index
// CHECK:           %[[VAL_30:.*]]:2 = "handshake.fork"(%[[VAL_28]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_31:.*]] = arith.cmpi slt, %[[VAL_30]]#1, %[[VAL_23]]#1 : index
// CHECK:           %[[VAL_32:.*]]:4 = "handshake.fork"(%[[VAL_31]]) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK:           %[[VAL_33:.*]], %[[VAL_34:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#3, %[[VAL_23]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_34]]) : (index) -> ()
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#2, %[[VAL_24]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_36]]) : (index) -> ()
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#1, %[[VAL_26]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#0, %[[VAL_30]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_40]]) : (index) -> ()
// CHECK:           %[[VAL_41:.*]] = "handshake.merge"(%[[VAL_39]]) : (index) -> index
// CHECK:           %[[VAL_42:.*]]:2 = "handshake.fork"(%[[VAL_41]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_43:.*]] = "handshake.merge"(%[[VAL_35]]) : (index) -> index
// CHECK:           %[[VAL_44:.*]] = "handshake.merge"(%[[VAL_33]]) : (index) -> index
// CHECK:           %[[VAL_45:.*]]:2 = "handshake.control_merge"(%[[VAL_37]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_46:.*]]:3 = "handshake.fork"(%[[VAL_45]]#0) {control = true} : (none) -> (none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_45]]#1) : (index) -> ()
// CHECK:           %[[VAL_47:.*]] = "handshake.constant"(%[[VAL_46]]#1) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_48:.*]] = "handshake.constant"(%[[VAL_46]]#0) {value = -10 : index} : (none) -> index
// CHECK:           %[[VAL_49:.*]] = arith.addi %[[VAL_42]]#1, %[[VAL_48]] : index
// CHECK:           %[[VAL_50:.*]] = arith.cmpi sge, %[[VAL_49]], %[[VAL_47]] : index
// CHECK:           %[[VAL_51:.*]]:4 = "handshake.fork"(%[[VAL_50]]) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = "handshake.conditional_branch"(%[[VAL_51]]#3, %[[VAL_42]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_54:.*]], %[[VAL_55:.*]] = "handshake.conditional_branch"(%[[VAL_51]]#2, %[[VAL_43]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_56:.*]], %[[VAL_57:.*]] = "handshake.conditional_branch"(%[[VAL_51]]#1, %[[VAL_44]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_58:.*]], %[[VAL_59:.*]] = "handshake.conditional_branch"(%[[VAL_51]]#0, %[[VAL_46]]#2) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_60:.*]] = "handshake.merge"(%[[VAL_52]]) : (index) -> index
// CHECK:           %[[VAL_61:.*]] = "handshake.merge"(%[[VAL_54]]) : (index) -> index
// CHECK:           %[[VAL_62:.*]] = "handshake.merge"(%[[VAL_56]]) : (index) -> index
// CHECK:           %[[VAL_63:.*]]:2 = "handshake.control_merge"(%[[VAL_58]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_63]]#1) : (index) -> ()
// CHECK:           %[[VAL_64:.*]] = "handshake.branch"(%[[VAL_60]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_65:.*]] = "handshake.branch"(%[[VAL_61]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_66:.*]] = "handshake.branch"(%[[VAL_62]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_67:.*]] = "handshake.branch"(%[[VAL_63]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_68:.*]] = "handshake.mux"(%[[VAL_69:.*]]#2, %[[VAL_64]], %[[VAL_53]]) : (index, index, index) -> index
// CHECK:           %[[VAL_70:.*]] = "handshake.mux"(%[[VAL_69]]#1, %[[VAL_65]], %[[VAL_55]]) : (index, index, index) -> index
// CHECK:           %[[VAL_71:.*]]:2 = "handshake.fork"(%[[VAL_70]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_72:.*]] = "handshake.mux"(%[[VAL_69]]#0, %[[VAL_66]], %[[VAL_57]]) : (index, index, index) -> index
// CHECK:           %[[VAL_73:.*]]:2 = "handshake.control_merge"(%[[VAL_67]], %[[VAL_59]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_69]]:3 = "handshake.fork"(%[[VAL_73]]#1) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_74:.*]] = arith.addi %[[VAL_68]], %[[VAL_71]]#1 : index
// CHECK:           %[[VAL_25]] = "handshake.branch"(%[[VAL_71]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_22]] = "handshake.branch"(%[[VAL_72]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_27]] = "handshake.branch"(%[[VAL_73]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_29]] = "handshake.branch"(%[[VAL_74]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_75:.*]]:2 = "handshake.control_merge"(%[[VAL_38]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_75]]#1) : (index) -> ()
// CHECK:           %[[VAL_76:.*]] = "handshake.branch"(%[[VAL_75]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_77:.*]]:2 = "handshake.control_merge"(%[[VAL_76]], %[[VAL_10]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_78:.*]]:4 = "handshake.fork"(%[[VAL_77]]#0) {control = true} : (none) -> (none, none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_77]]#1) : (index) -> ()
// CHECK:           %[[VAL_79:.*]] = "handshake.constant"(%[[VAL_78]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_80:.*]] = "handshake.constant"(%[[VAL_78]]#1) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_81:.*]] = "handshake.constant"(%[[VAL_78]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_82:.*]] = "handshake.branch"(%[[VAL_78]]#3) {control = true} : (none) -> none
// CHECK:           %[[VAL_83:.*]] = "handshake.branch"(%[[VAL_79]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_84:.*]] = "handshake.branch"(%[[VAL_80]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_85:.*]] = "handshake.branch"(%[[VAL_81]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_86:.*]] = "handshake.mux"(%[[VAL_87:.*]]#2, %[[VAL_88:.*]], %[[VAL_84]]) : (index, index, index) -> index
// CHECK:           %[[VAL_89:.*]]:2 = "handshake.fork"(%[[VAL_86]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_90:.*]] = "handshake.mux"(%[[VAL_87]]#1, %[[VAL_91:.*]], %[[VAL_85]]) : (index, index, index) -> index
// CHECK:           %[[VAL_92:.*]]:2 = "handshake.control_merge"(%[[VAL_93:.*]], %[[VAL_82]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_87]]:3 = "handshake.fork"(%[[VAL_92]]#1) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_94:.*]] = "handshake.mux"(%[[VAL_87]]#0, %[[VAL_95:.*]], %[[VAL_83]]) : (index, index, index) -> index
// CHECK:           %[[VAL_96:.*]]:2 = "handshake.fork"(%[[VAL_94]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_97:.*]] = arith.cmpi slt, %[[VAL_96]]#1, %[[VAL_89]]#1 : index
// CHECK:           %[[VAL_98:.*]]:4 = "handshake.fork"(%[[VAL_97]]) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK:           %[[VAL_99:.*]], %[[VAL_100:.*]] = "handshake.conditional_branch"(%[[VAL_98]]#3, %[[VAL_89]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_100]]) : (index) -> ()
// CHECK:           %[[VAL_101:.*]], %[[VAL_102:.*]] = "handshake.conditional_branch"(%[[VAL_98]]#2, %[[VAL_90]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_102]]) : (index) -> ()
// CHECK:           %[[VAL_103:.*]], %[[VAL_104:.*]] = "handshake.conditional_branch"(%[[VAL_98]]#1, %[[VAL_92]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_105:.*]], %[[VAL_106:.*]] = "handshake.conditional_branch"(%[[VAL_98]]#0, %[[VAL_96]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_106]]) : (index) -> ()
// CHECK:           %[[VAL_107:.*]] = "handshake.merge"(%[[VAL_105]]) : (index) -> index
// CHECK:           %[[VAL_108:.*]]:2 = "handshake.fork"(%[[VAL_107]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_109:.*]] = "handshake.merge"(%[[VAL_101]]) : (index) -> index
// CHECK:           %[[VAL_110:.*]] = "handshake.merge"(%[[VAL_99]]) : (index) -> index
// CHECK:           %[[VAL_111:.*]]:2 = "handshake.control_merge"(%[[VAL_103]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_112:.*]]:3 = "handshake.fork"(%[[VAL_111]]#0) {control = true} : (none) -> (none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_111]]#1) : (index) -> ()
// CHECK:           %[[VAL_113:.*]] = "handshake.constant"(%[[VAL_112]]#1) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_114:.*]] = "handshake.constant"(%[[VAL_112]]#0) {value = -10 : index} : (none) -> index
// CHECK:           %[[VAL_115:.*]] = arith.addi %[[VAL_108]]#1, %[[VAL_114]] : index
// CHECK:           %[[VAL_116:.*]] = arith.cmpi sge, %[[VAL_115]], %[[VAL_113]] : index
// CHECK:           %[[VAL_117:.*]]:4 = "handshake.fork"(%[[VAL_116]]) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK:           %[[VAL_118:.*]], %[[VAL_119:.*]] = "handshake.conditional_branch"(%[[VAL_117]]#3, %[[VAL_108]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_120:.*]], %[[VAL_121:.*]] = "handshake.conditional_branch"(%[[VAL_117]]#2, %[[VAL_109]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_122:.*]], %[[VAL_123:.*]] = "handshake.conditional_branch"(%[[VAL_117]]#1, %[[VAL_110]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_124:.*]], %[[VAL_125:.*]] = "handshake.conditional_branch"(%[[VAL_117]]#0, %[[VAL_112]]#2) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_126:.*]] = "handshake.merge"(%[[VAL_118]]) : (index) -> index
// CHECK:           %[[VAL_127:.*]] = "handshake.merge"(%[[VAL_120]]) : (index) -> index
// CHECK:           %[[VAL_128:.*]] = "handshake.merge"(%[[VAL_122]]) : (index) -> index
// CHECK:           %[[VAL_129:.*]]:2 = "handshake.control_merge"(%[[VAL_124]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_130:.*]]:4 = "handshake.fork"(%[[VAL_129]]#0) {control = true} : (none) -> (none, none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_129]]#1) : (index) -> ()
// CHECK:           %[[VAL_131:.*]] = "handshake.constant"(%[[VAL_130]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_132:.*]] = "handshake.constant"(%[[VAL_130]]#1) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_133:.*]] = "handshake.constant"(%[[VAL_130]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_134:.*]] = "handshake.branch"(%[[VAL_126]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_135:.*]] = "handshake.branch"(%[[VAL_127]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_136:.*]] = "handshake.branch"(%[[VAL_128]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_137:.*]] = "handshake.branch"(%[[VAL_130]]#3) {control = true} : (none) -> none
// CHECK:           %[[VAL_138:.*]] = "handshake.branch"(%[[VAL_131]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_139:.*]] = "handshake.branch"(%[[VAL_132]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_140:.*]] = "handshake.branch"(%[[VAL_133]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_141:.*]] = "handshake.mux"(%[[VAL_142:.*]]#5, %[[VAL_143:.*]], %[[VAL_139]]) : (index, index, index) -> index
// CHECK:           %[[VAL_144:.*]]:2 = "handshake.fork"(%[[VAL_141]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_145:.*]] = "handshake.mux"(%[[VAL_142]]#4, %[[VAL_146:.*]], %[[VAL_140]]) : (index, index, index) -> index
// CHECK:           %[[VAL_147:.*]] = "handshake.mux"(%[[VAL_142]]#3, %[[VAL_148:.*]], %[[VAL_134]]) : (index, index, index) -> index
// CHECK:           %[[VAL_149:.*]] = "handshake.mux"(%[[VAL_142]]#2, %[[VAL_150:.*]], %[[VAL_135]]) : (index, index, index) -> index
// CHECK:           %[[VAL_151:.*]] = "handshake.mux"(%[[VAL_142]]#1, %[[VAL_152:.*]], %[[VAL_136]]) : (index, index, index) -> index
// CHECK:           %[[VAL_153:.*]]:2 = "handshake.control_merge"(%[[VAL_154:.*]], %[[VAL_137]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_142]]:6 = "handshake.fork"(%[[VAL_153]]#1) {control = false} : (index) -> (index, index, index, index, index, index)
// CHECK:           %[[VAL_155:.*]] = "handshake.mux"(%[[VAL_142]]#0, %[[VAL_156:.*]], %[[VAL_138]]) : (index, index, index) -> index
// CHECK:           %[[VAL_157:.*]]:2 = "handshake.fork"(%[[VAL_155]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_158:.*]] = arith.cmpi slt, %[[VAL_157]]#1, %[[VAL_144]]#1 : index
// CHECK:           %[[VAL_159:.*]]:7 = "handshake.fork"(%[[VAL_158]]) {control = false} : (i1) -> (i1, i1, i1, i1, i1, i1, i1)
// CHECK:           %[[VAL_160:.*]], %[[VAL_161:.*]] = "handshake.conditional_branch"(%[[VAL_159]]#6, %[[VAL_144]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_161]]) : (index) -> ()
// CHECK:           %[[VAL_162:.*]], %[[VAL_163:.*]] = "handshake.conditional_branch"(%[[VAL_159]]#5, %[[VAL_145]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_163]]) : (index) -> ()
// CHECK:           %[[VAL_164:.*]], %[[VAL_165:.*]] = "handshake.conditional_branch"(%[[VAL_159]]#4, %[[VAL_147]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_166:.*]], %[[VAL_167:.*]] = "handshake.conditional_branch"(%[[VAL_159]]#3, %[[VAL_149]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_168:.*]], %[[VAL_169:.*]] = "handshake.conditional_branch"(%[[VAL_159]]#2, %[[VAL_151]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_170:.*]], %[[VAL_171:.*]] = "handshake.conditional_branch"(%[[VAL_159]]#1, %[[VAL_153]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_172:.*]], %[[VAL_173:.*]] = "handshake.conditional_branch"(%[[VAL_159]]#0, %[[VAL_157]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_173]]) : (index) -> ()
// CHECK:           %[[VAL_174:.*]] = "handshake.merge"(%[[VAL_172]]) : (index) -> index
// CHECK:           %[[VAL_175:.*]] = "handshake.merge"(%[[VAL_162]]) : (index) -> index
// CHECK:           %[[VAL_176:.*]]:2 = "handshake.fork"(%[[VAL_175]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_177:.*]] = "handshake.merge"(%[[VAL_160]]) : (index) -> index
// CHECK:           %[[VAL_178:.*]] = "handshake.merge"(%[[VAL_164]]) : (index) -> index
// CHECK:           %[[VAL_179:.*]] = "handshake.merge"(%[[VAL_166]]) : (index) -> index
// CHECK:           %[[VAL_180:.*]] = "handshake.merge"(%[[VAL_168]]) : (index) -> index
// CHECK:           %[[VAL_181:.*]]:2 = "handshake.control_merge"(%[[VAL_170]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_181]]#1) : (index) -> ()
// CHECK:           %[[VAL_182:.*]] = arith.addi %[[VAL_174]], %[[VAL_176]]#1 : index
// CHECK:           %[[VAL_146]] = "handshake.branch"(%[[VAL_176]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_143]] = "handshake.branch"(%[[VAL_177]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_148]] = "handshake.branch"(%[[VAL_178]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_150]] = "handshake.branch"(%[[VAL_179]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_152]] = "handshake.branch"(%[[VAL_180]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_154]] = "handshake.branch"(%[[VAL_181]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_156]] = "handshake.branch"(%[[VAL_182]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_183:.*]] = "handshake.merge"(%[[VAL_165]]) : (index) -> index
// CHECK:           %[[VAL_184:.*]] = "handshake.merge"(%[[VAL_167]]) : (index) -> index
// CHECK:           %[[VAL_185:.*]] = "handshake.merge"(%[[VAL_169]]) : (index) -> index
// CHECK:           %[[VAL_186:.*]]:2 = "handshake.control_merge"(%[[VAL_171]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_186]]#1) : (index) -> ()
// CHECK:           %[[VAL_187:.*]] = "handshake.branch"(%[[VAL_183]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_188:.*]] = "handshake.branch"(%[[VAL_184]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_189:.*]] = "handshake.branch"(%[[VAL_185]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_190:.*]] = "handshake.branch"(%[[VAL_186]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_191:.*]] = "handshake.mux"(%[[VAL_192:.*]]#2, %[[VAL_187]], %[[VAL_119]]) : (index, index, index) -> index
// CHECK:           %[[VAL_193:.*]] = "handshake.mux"(%[[VAL_192]]#1, %[[VAL_188]], %[[VAL_121]]) : (index, index, index) -> index
// CHECK:           %[[VAL_194:.*]]:2 = "handshake.fork"(%[[VAL_193]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_195:.*]] = "handshake.mux"(%[[VAL_192]]#0, %[[VAL_189]], %[[VAL_123]]) : (index, index, index) -> index
// CHECK:           %[[VAL_196:.*]]:2 = "handshake.control_merge"(%[[VAL_190]], %[[VAL_125]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_192]]:3 = "handshake.fork"(%[[VAL_196]]#1) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_197:.*]] = arith.addi %[[VAL_191]], %[[VAL_194]]#1 : index
// CHECK:           %[[VAL_91]] = "handshake.branch"(%[[VAL_194]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_88]] = "handshake.branch"(%[[VAL_195]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_93]] = "handshake.branch"(%[[VAL_196]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_95]] = "handshake.branch"(%[[VAL_197]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_198:.*]]:2 = "handshake.control_merge"(%[[VAL_104]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_198]]#1) : (index) -> ()
// CHECK:           handshake.return %[[VAL_198]]#0 : none
// CHECK:         }
// CHECK:       }

    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %1 = arith.muli %c-1, %c-1 : index
    %c20 = arith.constant 20 : index
    %2 = arith.addi %1, %c20 : index
    %3 = arith.cmpi sge, %2, %c0 : index
    cond_br %3, ^bb1, ^bb7
  ^bb1: // pred: ^bb0
    %c0_0 = arith.constant 0 : index
    %c42 = arith.constant 42 : index
    %c1 = arith.constant 1 : index
    br ^bb2(%c0_0 : index)
  ^bb2(%4: index):      // 2 preds: ^bb1, ^bb5
    %5 = arith.cmpi slt, %4, %c42 : index
    cond_br %5, ^bb3, ^bb6
  ^bb3: // pred: ^bb2
    %c0_1 = arith.constant 0 : index
    %c-10 = arith.constant -10 : index
    %6 = arith.addi %4, %c-10 : index
    %7 = arith.cmpi sge, %6, %c0_1 : index
    cond_br %7, ^bb4, ^bb5
  ^bb4: // pred: ^bb3
    br ^bb5
  ^bb5: // 2 preds: ^bb3, ^bb4
    %8 = arith.addi %4, %c1 : index
    br ^bb2(%8 : index)
  ^bb6: // pred: ^bb2
    br ^bb7
  ^bb7: // 2 preds: ^bb0, ^bb6
    %c0_2 = arith.constant 0 : index
    %c42_3 = arith.constant 42 : index
    %c1_4 = arith.constant 1 : index
    br ^bb8(%c0_2 : index)
  ^bb8(%9: index):      // 2 preds: ^bb7, ^bb14
    %10 = arith.cmpi slt, %9, %c42_3 : index
    cond_br %10, ^bb9, ^bb15
  ^bb9: // pred: ^bb8
    %c0_5 = arith.constant 0 : index
    %c-10_6 = arith.constant -10 : index
    %11 = arith.addi %9, %c-10_6 : index
    %12 = arith.cmpi sge, %11, %c0_5 : index
    cond_br %12, ^bb10, ^bb14
  ^bb10:        // pred: ^bb9
    %c0_7 = arith.constant 0 : index
    %c42_8 = arith.constant 42 : index
    %c1_9 = arith.constant 1 : index
    br ^bb11(%c0_7 : index)
  ^bb11(%13: index):    // 2 preds: ^bb10, ^bb12
    %14 = arith.cmpi slt, %13, %c42_8 : index
    cond_br %14, ^bb12, ^bb13
  ^bb12:        // pred: ^bb11
    %15 = arith.addi %13, %c1_9 : index
    br ^bb11(%15 : index)
  ^bb13:        // pred: ^bb11
    br ^bb14
  ^bb14:        // 2 preds: ^bb9, ^bb13
    %16 = arith.addi %9, %c1_4 : index
    br ^bb8(%16 : index)
  ^bb15:        // pred: ^bb8
    return
  }
