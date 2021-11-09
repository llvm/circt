// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
func @loop_min_max(%arg0: index) {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @loop_min_max(
// CHECK-SAME:                                 %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = "handshake.merge"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_3:.*]]:4 = "handshake.fork"(%[[VAL_1]]) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_4:.*]] = "handshake.constant"(%[[VAL_3]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_5:.*]] = "handshake.constant"(%[[VAL_3]]#1) {value = 42 : index} : (none) -> index
// CHECK:           %[[VAL_6:.*]] = "handshake.constant"(%[[VAL_3]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_7:.*]] = "handshake.branch"(%[[VAL_2]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_8:.*]] = "handshake.branch"(%[[VAL_3]]#3) {control = true} : (none) -> none
// CHECK:           %[[VAL_9:.*]] = "handshake.branch"(%[[VAL_4]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_10:.*]] = "handshake.branch"(%[[VAL_5]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_11:.*]] = "handshake.branch"(%[[VAL_6]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_12:.*]] = "handshake.mux"(%[[VAL_13:.*]]#3, %[[VAL_14:.*]], %[[VAL_10]]) : (index, index, index) -> index
// CHECK:           %[[VAL_15:.*]]:2 = "handshake.fork"(%[[VAL_12]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_16:.*]] = "handshake.mux"(%[[VAL_13]]#2, %[[VAL_17:.*]], %[[VAL_7]]) : (index, index, index) -> index
// CHECK:           %[[VAL_18:.*]] = "handshake.mux"(%[[VAL_13]]#1, %[[VAL_19:.*]], %[[VAL_11]]) : (index, index, index) -> index
// CHECK:           %[[VAL_20:.*]]:2 = "handshake.control_merge"(%[[VAL_21:.*]], %[[VAL_8]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_13]]:4 = "handshake.fork"(%[[VAL_20]]#1) {control = false} : (index) -> (index, index, index, index)
// CHECK:           %[[VAL_22:.*]] = "handshake.mux"(%[[VAL_13]]#0, %[[VAL_23:.*]], %[[VAL_9]]) : (index, index, index) -> index
// CHECK:           %[[VAL_24:.*]]:2 = "handshake.fork"(%[[VAL_22]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_24]]#1, %[[VAL_15]]#1 : index
// CHECK:           %[[VAL_26:.*]]:5 = "handshake.fork"(%[[VAL_25]]) {control = false} : (i1) -> (i1, i1, i1, i1, i1)
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = "handshake.conditional_branch"(%[[VAL_26]]#4, %[[VAL_15]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_28]]) : (index) -> ()
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]] = "handshake.conditional_branch"(%[[VAL_26]]#3, %[[VAL_16]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_30]]) : (index) -> ()
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = "handshake.conditional_branch"(%[[VAL_26]]#2, %[[VAL_18]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_32]]) : (index) -> ()
// CHECK:           %[[VAL_33:.*]], %[[VAL_34:.*]] = "handshake.conditional_branch"(%[[VAL_26]]#1, %[[VAL_20]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = "handshake.conditional_branch"(%[[VAL_26]]#0, %[[VAL_24]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_36]]) : (index) -> ()
// CHECK:           %[[VAL_37:.*]] = "handshake.merge"(%[[VAL_35]]) : (index) -> index
// CHECK:           %[[VAL_38:.*]]:5 = "handshake.fork"(%[[VAL_37]]) {control = false} : (index) -> (index, index, index, index, index)
// CHECK:           %[[VAL_39:.*]] = "handshake.merge"(%[[VAL_29]]) : (index) -> index
// CHECK:           %[[VAL_40:.*]]:4 = "handshake.fork"(%[[VAL_39]]) {control = false} : (index) -> (index, index, index, index)
// CHECK:           %[[VAL_41:.*]] = "handshake.merge"(%[[VAL_31]]) : (index) -> index
// CHECK:           %[[VAL_42:.*]] = "handshake.merge"(%[[VAL_27]]) : (index) -> index
// CHECK:           %[[VAL_43:.*]]:2 = "handshake.control_merge"(%[[VAL_33]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_44:.*]]:4 = "handshake.fork"(%[[VAL_43]]#0) {control = true} : (none) -> (none, none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_43]]#1) : (index) -> ()
// CHECK:           %[[VAL_45:.*]] = "handshake.constant"(%[[VAL_44]]#2) {value = -1 : index} : (none) -> index
// CHECK:           %[[VAL_46:.*]] = arith.muli %[[VAL_38]]#4, %[[VAL_45]] : index
// CHECK:           %[[VAL_47:.*]] = arith.addi %[[VAL_46]], %[[VAL_40]]#3 : index
// CHECK:           %[[VAL_48:.*]]:2 = "handshake.fork"(%[[VAL_47]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_49:.*]] = arith.cmpi sgt, %[[VAL_38]]#3, %[[VAL_48]]#1 : index
// CHECK:           %[[VAL_50:.*]] = select %[[VAL_49]], %[[VAL_38]]#2, %[[VAL_48]]#0 : index
// CHECK:           %[[VAL_51:.*]] = "handshake.constant"(%[[VAL_44]]#1) {value = 10 : index} : (none) -> index
// CHECK:           %[[VAL_52:.*]] = arith.addi %[[VAL_38]]#1, %[[VAL_51]] : index
// CHECK:           %[[VAL_53:.*]]:2 = "handshake.fork"(%[[VAL_52]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_54:.*]] = arith.cmpi slt, %[[VAL_40]]#2, %[[VAL_53]]#1 : index
// CHECK:           %[[VAL_55:.*]] = select %[[VAL_54]], %[[VAL_40]]#1, %[[VAL_53]]#0 : index
// CHECK:           %[[VAL_56:.*]] = "handshake.constant"(%[[VAL_44]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_57:.*]] = "handshake.branch"(%[[VAL_38]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_58:.*]] = "handshake.branch"(%[[VAL_40]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_59:.*]] = "handshake.branch"(%[[VAL_41]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_60:.*]] = "handshake.branch"(%[[VAL_42]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_61:.*]] = "handshake.branch"(%[[VAL_44]]#3) {control = true} : (none) -> none
// CHECK:           %[[VAL_62:.*]] = "handshake.branch"(%[[VAL_50]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_63:.*]] = "handshake.branch"(%[[VAL_55]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_64:.*]] = "handshake.branch"(%[[VAL_56]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_65:.*]] = "handshake.mux"(%[[VAL_66:.*]]#6, %[[VAL_67:.*]], %[[VAL_63]]) : (index, index, index) -> index
// CHECK:           %[[VAL_68:.*]]:2 = "handshake.fork"(%[[VAL_65]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_69:.*]] = "handshake.mux"(%[[VAL_66]]#5, %[[VAL_70:.*]], %[[VAL_64]]) : (index, index, index) -> index
// CHECK:           %[[VAL_71:.*]] = "handshake.mux"(%[[VAL_66]]#4, %[[VAL_72:.*]], %[[VAL_57]]) : (index, index, index) -> index
// CHECK:           %[[VAL_73:.*]] = "handshake.mux"(%[[VAL_66]]#3, %[[VAL_74:.*]], %[[VAL_59]]) : (index, index, index) -> index
// CHECK:           %[[VAL_75:.*]] = "handshake.mux"(%[[VAL_66]]#2, %[[VAL_76:.*]], %[[VAL_60]]) : (index, index, index) -> index
// CHECK:           %[[VAL_77:.*]] = "handshake.mux"(%[[VAL_66]]#1, %[[VAL_78:.*]], %[[VAL_58]]) : (index, index, index) -> index
// CHECK:           %[[VAL_79:.*]]:2 = "handshake.control_merge"(%[[VAL_80:.*]], %[[VAL_61]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_66]]:7 = "handshake.fork"(%[[VAL_79]]#1) {control = false} : (index) -> (index, index, index, index, index, index, index)
// CHECK:           %[[VAL_81:.*]] = "handshake.mux"(%[[VAL_66]]#0, %[[VAL_82:.*]], %[[VAL_62]]) : (index, index, index) -> index
// CHECK:           %[[VAL_83:.*]]:2 = "handshake.fork"(%[[VAL_81]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_84:.*]] = arith.cmpi slt, %[[VAL_83]]#1, %[[VAL_68]]#1 : index
// CHECK:           %[[VAL_85:.*]]:8 = "handshake.fork"(%[[VAL_84]]) {control = false} : (i1) -> (i1, i1, i1, i1, i1, i1, i1, i1)
// CHECK:           %[[VAL_86:.*]], %[[VAL_87:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#7, %[[VAL_68]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_87]]) : (index) -> ()
// CHECK:           %[[VAL_88:.*]], %[[VAL_89:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#6, %[[VAL_69]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_89]]) : (index) -> ()
// CHECK:           %[[VAL_90:.*]], %[[VAL_91:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#5, %[[VAL_71]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_92:.*]], %[[VAL_93:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#4, %[[VAL_73]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_94:.*]], %[[VAL_95:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#3, %[[VAL_75]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_96:.*]], %[[VAL_97:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#2, %[[VAL_77]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           %[[VAL_98:.*]], %[[VAL_99:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#1, %[[VAL_79]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_100:.*]], %[[VAL_101:.*]] = "handshake.conditional_branch"(%[[VAL_85]]#0, %[[VAL_83]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_101]]) : (index) -> ()
// CHECK:           %[[VAL_102:.*]] = "handshake.merge"(%[[VAL_100]]) : (index) -> index
// CHECK:           %[[VAL_103:.*]] = "handshake.merge"(%[[VAL_88]]) : (index) -> index
// CHECK:           %[[VAL_104:.*]]:2 = "handshake.fork"(%[[VAL_103]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_105:.*]] = "handshake.merge"(%[[VAL_86]]) : (index) -> index
// CHECK:           %[[VAL_106:.*]] = "handshake.merge"(%[[VAL_90]]) : (index) -> index
// CHECK:           %[[VAL_107:.*]] = "handshake.merge"(%[[VAL_92]]) : (index) -> index
// CHECK:           %[[VAL_108:.*]] = "handshake.merge"(%[[VAL_94]]) : (index) -> index
// CHECK:           %[[VAL_109:.*]] = "handshake.merge"(%[[VAL_96]]) : (index) -> index
// CHECK:           %[[VAL_110:.*]]:2 = "handshake.control_merge"(%[[VAL_98]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_110]]#1) : (index) -> ()
// CHECK:           %[[VAL_111:.*]] = arith.addi %[[VAL_102]], %[[VAL_104]]#1 : index
// CHECK:           %[[VAL_70]] = "handshake.branch"(%[[VAL_104]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_67]] = "handshake.branch"(%[[VAL_105]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_72]] = "handshake.branch"(%[[VAL_106]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_74]] = "handshake.branch"(%[[VAL_107]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_76]] = "handshake.branch"(%[[VAL_108]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_78]] = "handshake.branch"(%[[VAL_109]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_80]] = "handshake.branch"(%[[VAL_110]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_82]] = "handshake.branch"(%[[VAL_111]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_112:.*]] = "handshake.merge"(%[[VAL_91]]) : (index) -> index
// CHECK:           %[[VAL_113:.*]] = "handshake.merge"(%[[VAL_93]]) : (index) -> index
// CHECK:           %[[VAL_114:.*]]:2 = "handshake.fork"(%[[VAL_113]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_115:.*]] = "handshake.merge"(%[[VAL_95]]) : (index) -> index
// CHECK:           %[[VAL_116:.*]] = "handshake.merge"(%[[VAL_97]]) : (index) -> index
// CHECK:           %[[VAL_117:.*]]:2 = "handshake.control_merge"(%[[VAL_99]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_117]]#1) : (index) -> ()
// CHECK:           %[[VAL_118:.*]] = arith.addi %[[VAL_112]], %[[VAL_114]]#1 : index
// CHECK:           %[[VAL_19]] = "handshake.branch"(%[[VAL_114]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_14]] = "handshake.branch"(%[[VAL_115]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_17]] = "handshake.branch"(%[[VAL_116]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_21]] = "handshake.branch"(%[[VAL_117]]#0) {control = true} : (none) -> none
// CHECK:           %[[VAL_23]] = "handshake.branch"(%[[VAL_118]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_119:.*]]:2 = "handshake.control_merge"(%[[VAL_34]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_119]]#1) : (index) -> ()
// CHECK:           return %[[VAL_119]]#0 : none
// CHECK:         }
// CHECK:       }

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
