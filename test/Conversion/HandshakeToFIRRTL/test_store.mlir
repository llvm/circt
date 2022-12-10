// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK:           firrtl.module @handshake_store_in_ui64_ui8_out_ui8_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_9:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_10:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_12:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_13:.*]] = firrtl.subfield %[[VAL_3]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_14:.*]] = firrtl.subfield %[[VAL_3]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_15:.*]] = firrtl.subfield %[[VAL_3]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_16:.*]] = firrtl.subfield %[[VAL_4]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_17:.*]] = firrtl.subfield %[[VAL_4]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_18:.*]] = firrtl.subfield %[[VAL_4]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             %[[VAL_19:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:             %[[VAL_20:.*]] = firrtl.and %[[VAL_14]], %[[VAL_17]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_21:.*]] = firrtl.and %[[VAL_5]], %[[VAL_8]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_22:.*]] = firrtl.and %[[VAL_11]], %[[VAL_21]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_19]], %[[VAL_22]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_23:.*]] = firrtl.and %[[VAL_20]], %[[VAL_22]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_9]], %[[VAL_23]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_6]], %[[VAL_23]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_12]], %[[VAL_23]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_18]], %[[VAL_7]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:             firrtl.connect %[[VAL_15]], %[[VAL_10]] : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK:             firrtl.connect %[[VAL_13]], %[[VAL_19]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_16]], %[[VAL_19]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }

// CHECK: firrtl.module @main(in %[[VAL_24:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in %[[VAL_25:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_26:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_27:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %[[VAL_28:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_29:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_30:.*]]: !firrtl.clock, in %[[VAL_31:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_32:.*]], %[[VAL_33:.*]], %[[VAL_34:.*]], %[[VAL_35:.*]], %[[VAL_36:.*]] = firrtl.instance handshake_store0  @handshake_store_in_ui64_ui8_out_ui8_ui64(in addrIn0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in dataIn: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out dataToMem: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out addrOut0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @main(%arg0: i8, %arg1: index, %arg2: none, ...) -> (i8, index, none) {
  %0:2 = fork [2] %arg2 : none
  %1:2 = store [%arg1] %arg0, %0#0 : index, i8
  return %1#0, %1#1, %0#1 : i8, index, none
}
