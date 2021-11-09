// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
  func @affine_dma_start(%arg0: index) {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @affine_dma_start(
// CHECK-SAME:                                     %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = "handshake.merge"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_3:.*]]:6 = "handshake.fork"(%[[VAL_1]]) {control = true} : (none) -> (none, none, none, none, none, none)
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<100xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<100xf32, 2>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_7:.*]] = "handshake.constant"(%[[VAL_3]]#4) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_8:.*]] = "handshake.constant"(%[[VAL_3]]#3) {value = 64 : index} : (none) -> index
// CHECK:           %[[VAL_9:.*]] = "handshake.constant"(%[[VAL_3]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_10:.*]] = "handshake.constant"(%[[VAL_3]]#1) {value = 10 : index} : (none) -> index
// CHECK:           %[[VAL_11:.*]] = "handshake.constant"(%[[VAL_3]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_12:.*]] = "handshake.branch"(%[[VAL_2]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_13:.*]] = "handshake.branch"(%[[VAL_3]]#5) {control = true} : (none) -> none
// CHECK:           %[[VAL_14:.*]] = "handshake.branch"(%[[VAL_4]]) {control = false} : (memref<100xf32>) -> memref<100xf32>
// CHECK:           %[[VAL_15:.*]] = "handshake.branch"(%[[VAL_5]]) {control = false} : (memref<100xf32, 2>) -> memref<100xf32, 2>
// CHECK:           %[[VAL_16:.*]] = "handshake.branch"(%[[VAL_6]]) {control = false} : (memref<1xi32>) -> memref<1xi32>
// CHECK:           %[[VAL_17:.*]] = "handshake.branch"(%[[VAL_7]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_18:.*]] = "handshake.branch"(%[[VAL_8]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_19:.*]] = "handshake.branch"(%[[VAL_9]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_20:.*]] = "handshake.branch"(%[[VAL_10]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_21:.*]] = "handshake.branch"(%[[VAL_11]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_22:.*]] = "handshake.mux"(%[[VAL_23:.*]]#8, %[[VAL_24:.*]], %[[VAL_20]]) : (index, index, index) -> index
// CHECK:           %[[VAL_25:.*]]:2 = "handshake.fork"(%[[VAL_22]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_26:.*]] = "handshake.mux"(%[[VAL_23]]#7, %[[VAL_27:.*]], %[[VAL_12]]) : (index, index, index) -> index
// CHECK:           %[[VAL_28:.*]] = "handshake.mux"(%[[VAL_23]]#6, %[[VAL_29:.*]], %[[VAL_14]]) : (index, memref<100xf32>, memref<100xf32>) -> memref<100xf32>
// CHECK:           %[[VAL_30:.*]] = "handshake.mux"(%[[VAL_23]]#5, %[[VAL_31:.*]], %[[VAL_15]]) : (index, memref<100xf32, 2>, memref<100xf32, 2>) -> memref<100xf32, 2>
// CHECK:           %[[VAL_32:.*]] = "handshake.mux"(%[[VAL_23]]#4, %[[VAL_33:.*]], %[[VAL_18]]) : (index, index, index) -> index
// CHECK:           %[[VAL_34:.*]] = "handshake.mux"(%[[VAL_23]]#3, %[[VAL_35:.*]], %[[VAL_16]]) : (index, memref<1xi32>, memref<1xi32>) -> memref<1xi32>
// CHECK:           %[[VAL_36:.*]] = "handshake.mux"(%[[VAL_23]]#2, %[[VAL_37:.*]], %[[VAL_17]]) : (index, index, index) -> index
// CHECK:           %[[VAL_38:.*]] = "handshake.mux"(%[[VAL_23]]#1, %[[VAL_39:.*]], %[[VAL_21]]) : (index, index, index) -> index
// CHECK:           %[[VAL_40:.*]]:2 = "handshake.control_merge"(%[[VAL_41:.*]], %[[VAL_13]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_23]]:9 = "handshake.fork"(%[[VAL_40]]#1) {control = false} : (index) -> (index, index, index, index, index, index, index, index, index)
// CHECK:           %[[VAL_42:.*]] = "handshake.mux"(%[[VAL_23]]#0, %[[VAL_43:.*]], %[[VAL_19]]) : (index, index, index) -> index
// CHECK:           %[[VAL_44:.*]]:2 = "handshake.fork"(%[[VAL_42]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_44]]#1, %[[VAL_25]]#1 : index
// CHECK:           %[[VAL_46:.*]]:10 = "handshake.fork"(%[[VAL_45]]) {control = false} : (i1) -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1)
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#9, %[[VAL_25]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_48]]) : (index) -> ()
// CHECK:           %[[VAL_49:.*]], %[[VAL_50:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#8, %[[VAL_26]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_50]]) : (index) -> ()
// CHECK:           %[[VAL_51:.*]], %[[VAL_52:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#7, %[[VAL_28]]) {control = false} : (i1, memref<100xf32>) -> (memref<100xf32>, memref<100xf32>)
// CHECK:           "handshake.sink"(%[[VAL_52]]) : (memref<100xf32>) -> ()
// CHECK:           %[[VAL_53:.*]], %[[VAL_54:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#6, %[[VAL_30]]) {control = false} : (i1, memref<100xf32, 2>) -> (memref<100xf32, 2>, memref<100xf32, 2>)
// CHECK:           "handshake.sink"(%[[VAL_54]]) : (memref<100xf32, 2>) -> ()
// CHECK:           %[[VAL_55:.*]], %[[VAL_56:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#5, %[[VAL_32]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_56]]) : (index) -> ()
// CHECK:           %[[VAL_57:.*]], %[[VAL_58:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#4, %[[VAL_34]]) {control = false} : (i1, memref<1xi32>) -> (memref<1xi32>, memref<1xi32>)
// CHECK:           "handshake.sink"(%[[VAL_58]]) : (memref<1xi32>) -> ()
// CHECK:           %[[VAL_59:.*]], %[[VAL_60:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#3, %[[VAL_36]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_60]]) : (index) -> ()
// CHECK:           %[[VAL_61:.*]], %[[VAL_62:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#2, %[[VAL_38]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_62]]) : (index) -> ()
// CHECK:           %[[VAL_63:.*]], %[[VAL_64:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#1, %[[VAL_40]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_65:.*]], %[[VAL_66:.*]] = "handshake.conditional_branch"(%[[VAL_46]]#0, %[[VAL_44]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_66]]) : (index) -> ()
// CHECK:           %[[VAL_67:.*]] = "handshake.merge"(%[[VAL_65]]) : (index) -> index
// CHECK:           %[[VAL_68:.*]]:2 = "handshake.fork"(%[[VAL_67]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_69:.*]] = "handshake.merge"(%[[VAL_49]]) : (index) -> index
// CHECK:           %[[VAL_70:.*]]:2 = "handshake.fork"(%[[VAL_69]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_71:.*]] = "handshake.merge"(%[[VAL_51]]) : (memref<100xf32>) -> memref<100xf32>
// CHECK:           %[[VAL_72:.*]]:2 = "handshake.fork"(%[[VAL_71]]) {control = false} : (memref<100xf32>) -> (memref<100xf32>, memref<100xf32>)
// CHECK:           %[[VAL_73:.*]] = "handshake.merge"(%[[VAL_53]]) : (memref<100xf32, 2>) -> memref<100xf32, 2>
// CHECK:           %[[VAL_74:.*]]:2 = "handshake.fork"(%[[VAL_73]]) {control = false} : (memref<100xf32, 2>) -> (memref<100xf32, 2>, memref<100xf32, 2>)
// CHECK:           %[[VAL_75:.*]] = "handshake.merge"(%[[VAL_55]]) : (index) -> index
// CHECK:           %[[VAL_76:.*]]:2 = "handshake.fork"(%[[VAL_75]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_77:.*]] = "handshake.merge"(%[[VAL_57]]) : (memref<1xi32>) -> memref<1xi32>
// CHECK:           %[[VAL_78:.*]]:2 = "handshake.fork"(%[[VAL_77]]) {control = false} : (memref<1xi32>) -> (memref<1xi32>, memref<1xi32>)
// CHECK:           %[[VAL_79:.*]] = "handshake.merge"(%[[VAL_59]]) : (index) -> index
// CHECK:           %[[VAL_80:.*]]:2 = "handshake.fork"(%[[VAL_79]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_81:.*]] = "handshake.merge"(%[[VAL_61]]) : (index) -> index
// CHECK:           %[[VAL_82:.*]]:2 = "handshake.fork"(%[[VAL_81]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_83:.*]] = "handshake.merge"(%[[VAL_47]]) : (index) -> index
// CHECK:           %[[VAL_84:.*]]:2 = "handshake.control_merge"(%[[VAL_63]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_85:.*]]:3 = "handshake.fork"(%[[VAL_84]]#0) {control = true} : (none) -> (none, none, none)
// CHECK:           "handshake.sink"(%[[VAL_84]]#1) : (index) -> ()
// CHECK:           %[[VAL_86:.*]] = "handshake.constant"(%[[VAL_85]]#1) {value = 7 : index} : (none) -> index
// CHECK:           %[[VAL_87:.*]] = arith.addi %[[VAL_68]]#1, %[[VAL_86]] : index
// CHECK:           %[[VAL_88:.*]] = "handshake.constant"(%[[VAL_85]]#0) {value = 11 : index} : (none) -> index
// CHECK:           %[[VAL_89:.*]] = arith.addi %[[VAL_70]]#1, %[[VAL_88]] : index
// CHECK:           memref.dma_start %[[VAL_72]]#1{{\[}}%[[VAL_87]]], %[[VAL_74]]#1{{\[}}%[[VAL_89]]], %[[VAL_76]]#1, %[[VAL_78]]#1{{\[}}%[[VAL_80]]#1] : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
// CHECK:           %[[VAL_90:.*]] = arith.addi %[[VAL_68]]#0, %[[VAL_82]]#1 : index
// CHECK:           %[[VAL_27]] = "handshake.branch"(%[[VAL_70]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_29]] = "handshake.branch"(%[[VAL_72]]#0) {control = false} : (memref<100xf32>) -> memref<100xf32>
// CHECK:           %[[VAL_31]] = "handshake.branch"(%[[VAL_74]]#0) {control = false} : (memref<100xf32, 2>) -> memref<100xf32, 2>
// CHECK:           %[[VAL_33]] = "handshake.branch"(%[[VAL_76]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_35]] = "handshake.branch"(%[[VAL_78]]#0) {control = false} : (memref<1xi32>) -> memref<1xi32>
// CHECK:           %[[VAL_37]] = "handshake.branch"(%[[VAL_80]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_39]] = "handshake.branch"(%[[VAL_82]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_24]] = "handshake.branch"(%[[VAL_83]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_41]] = "handshake.branch"(%[[VAL_85]]#2) {control = true} : (none) -> none
// CHECK:           %[[VAL_43]] = "handshake.branch"(%[[VAL_90]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_91:.*]]:2 = "handshake.control_merge"(%[[VAL_64]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_91]]#1) : (index) -> ()
// CHECK:           return %[[VAL_91]]#0 : none
// CHECK:         }
// CHECK:       }

    %0 = memref.alloc() : memref<100xf32>
    %1 = memref.alloc() : memref<100xf32, 2>
    %2 = memref.alloc() : memref<1xi32>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c0_0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    br ^bb1(%c0_0 : index)
  ^bb1(%3: index):      // 2 preds: ^bb0, ^bb2
    %4 = arith.cmpi slt, %3, %c10 : index
    cond_br %4, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    %c7 = arith.constant 7 : index
    %5 = arith.addi %3, %c7 : index
    %c11 = arith.constant 11 : index
    %6 = arith.addi %arg0, %c11 : index
    memref.dma_start %0[%5], %1[%6], %c64, %2[%c0] : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
    %7 = arith.addi %3, %c1 : index
    br ^bb1(%7 : index)
  ^bb3: // pred: ^bb1
    return
  }
