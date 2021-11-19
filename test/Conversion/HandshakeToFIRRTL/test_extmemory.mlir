// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK:           firrtl.module @handshake_extmemory_in_ui32_ui64_ui64_out_ui32(in %[[VAL_0:.*]]: !firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_5:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_6:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_0]](0) : (!firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_0]](1) : (!firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_9:.*]] = firrtl.subfield %[[VAL_0]](2) : (!firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_10:.*]] = firrtl.subfield %[[VAL_0]](3) : (!firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_11:.*]] = firrtl.subfield %[[VAL_0]](4) : (!firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_12:.*]] = firrtl.subfield %[[VAL_0]](5) : (!firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>) -> !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_13:.*]] = firrtl.subfield %[[VAL_1]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_14:.*]] = firrtl.subfield %[[VAL_1]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_15:.*]] = firrtl.subfield %[[VAL_1]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<32>
// CHECK:             %[[VAL_16:.*]] = firrtl.subfield %[[VAL_2]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_17:.*]] = firrtl.subfield %[[VAL_2]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_18:.*]] = firrtl.subfield %[[VAL_2]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:             %[[VAL_19:.*]] = firrtl.subfield %[[VAL_3]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_20:.*]] = firrtl.subfield %[[VAL_3]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_21:.*]] = firrtl.subfield %[[VAL_3]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:             %[[VAL_22:.*]] = firrtl.subfield %[[VAL_4]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_23:.*]] = firrtl.subfield %[[VAL_4]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_24:.*]] = firrtl.subfield %[[VAL_4]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<32>
// CHECK:             %[[VAL_25:.*]] = firrtl.subfield %[[VAL_5]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_26:.*]] = firrtl.subfield %[[VAL_5]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_27:.*]] = firrtl.subfield %[[VAL_6]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_28:.*]] = firrtl.subfield %[[VAL_6]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_29:.*]] = firrtl.subfield %[[VAL_7]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_29]], %[[VAL_13]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_30:.*]] = firrtl.subfield %[[VAL_7]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_14]], %[[VAL_30]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_31:.*]] = firrtl.subfield %[[VAL_7]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<32>
// CHECK:             firrtl.connect %[[VAL_31]], %[[VAL_15]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:             %[[VAL_32:.*]] = firrtl.subfield %[[VAL_8]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_32]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_33:.*]] = firrtl.subfield %[[VAL_8]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_17]], %[[VAL_33]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_34:.*]] = firrtl.subfield %[[VAL_8]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:             firrtl.connect %[[VAL_34]], %[[VAL_18]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:             %[[VAL_35:.*]] = firrtl.subfield %[[VAL_9]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_35]], %[[VAL_19]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_36:.*]] = firrtl.subfield %[[VAL_9]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_20]], %[[VAL_36]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_37:.*]] = firrtl.subfield %[[VAL_9]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:             firrtl.connect %[[VAL_37]], %[[VAL_21]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:             %[[VAL_38:.*]] = firrtl.subfield %[[VAL_10]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_22]], %[[VAL_38]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_39:.*]] = firrtl.subfield %[[VAL_10]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_39]], %[[VAL_23]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_40:.*]] = firrtl.subfield %[[VAL_10]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) -> !firrtl.uint<32>
// CHECK:             firrtl.connect %[[VAL_24]], %[[VAL_40]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:             %[[VAL_41:.*]] = firrtl.subfield %[[VAL_11]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_25]], %[[VAL_41]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_42:.*]] = firrtl.subfield %[[VAL_11]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_42]], %[[VAL_26]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_43:.*]] = firrtl.subfield %[[VAL_12]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_27]], %[[VAL_43]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_44:.*]] = firrtl.subfield %[[VAL_12]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_44]], %[[VAL_28]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }

// CHECK:           firrtl.module @main(in %[[VAL_138:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_139:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_140:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_141:.*]]: !firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>, in %[[VAL_142:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_143:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_144:.*]]: !firrtl.clock, in %[[VAL_145:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_146:.*]], %[[VAL_147:.*]], %[[VAL_148:.*]], %[[VAL_149:.*]], %[[VAL_150:.*]], %[[VAL_151:.*]], %[[VAL_152:.*]] = firrtl.instance handshake_extmemory0  @handshake_extmemory_in_ui32_ui64_ui64_out_ui32(in [[ARG0:.+]]: !firrtl.bundle<[[ARG1:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG2:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG3:.+]] flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, [[ARG4:.+]]: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, [[ARG5:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>, [[ARG6:.+]]: bundle<valid: uint<1>, ready flip: uint<1>>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in [[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG4:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out [[ARG5:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG6:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)

handshake.func @main(%arg0: index, %arg1: index, %v: i32, %mem : memref<10xi32>, %argCtrl: none) -> none {
  %ldData, %stCtrl, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr, %loadAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%arg0] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%arg1] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none
  return %finCtrl : none
}
