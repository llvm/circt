// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file -verify-diagnostics %s

// CHECK:           firrtl.module @handshake_load_in_ui64_ui8_out_ui8_ui64(in %[[VAL_87:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_88:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in %[[VAL_89:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_90:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %[[VAL_91:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:             %[[VAL_92:.*]] = firrtl.subfield %[[VAL_87]][0] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_93:.*]] = firrtl.subfield %[[VAL_87]][1] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_94:.*]] = firrtl.subfield %[[VAL_87]][2] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_95:.*]] = firrtl.subfield %[[VAL_88]][0] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_96:.*]] = firrtl.subfield %[[VAL_88]][1] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_97:.*]] = firrtl.subfield %[[VAL_88]][2] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_98:.*]] = firrtl.subfield %[[VAL_89]][0] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_99:.*]] = firrtl.subfield %[[VAL_89]][1] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_100:.*]] = firrtl.subfield %[[VAL_90]][0] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_101:.*]] = firrtl.subfield %[[VAL_90]][1] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_102:.*]] = firrtl.subfield %[[VAL_90]][2] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_103:.*]] = firrtl.subfield %[[VAL_91]][0] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_104:.*]] = firrtl.subfield %[[VAL_91]][1] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_105:.*]] = firrtl.subfield %[[VAL_91]][2] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             firrtl.connect %[[VAL_105]], %[[VAL_94]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:             firrtl.connect %[[VAL_102]], %[[VAL_97]] : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK:             %[[VAL_106:.*]] = firrtl.and %[[VAL_92]], %[[VAL_98]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_103]], %[[VAL_106]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_107:.*]] = firrtl.and %[[VAL_106]], %[[VAL_104]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_93]], %[[VAL_107]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_99]], %[[VAL_107]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_100]], %[[VAL_95]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_96]], %[[VAL_101]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }

// CHECK:           firrtl.module @main(in %[[VAL_111:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_112:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_113:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_114:.*]]: !firrtl.clock, in %[[VAL_115:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_116:.*]], %[[VAL_117:.*]], %[[VAL_118:.*]], %[[VAL_119:.*]], %[[VAL_120:.*]] = firrtl.instance handshake_memory0  @handshake_memory_out_ui8_id0(in ldAddr0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out lddata0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out ldDone0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
handshake.func @main(%arg0: index, %arg1: none, ...) -> none {
  %0:2 = memory [ld = 1, st= 0] (%addressResults) {id = 0 : i32, lsq = false} : memref<10xi8>, (index) -> (i8, none)
  %1:2 = fork [2] %arg1 : none
  %2 = join %1#1, %0#1 : none, none
  %3, %addressResults = load [%arg0] %0#0, %1#0 : index, i8
  sink %3 : i8
  return %2 : none
}
