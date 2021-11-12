// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK:           firrtl.module @handshake_store_in_ui8_ui64_out_ui8_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK: %[[IN_DATA_VALID:.+]] = firrtl.subfield %[[VAL_0]](0)
// CHECK: %[[IN_DATA_READY:.+]] = firrtl.subfield %[[VAL_0]](1)
// CHECK: %[[IN_DATA_DATA:.+]] = firrtl.subfield %[[VAL_0]](2)
// CHECK: %[[IN_ADDR_VALID:.+]] = firrtl.subfield %[[VAL_1]](0)
// CHECK: %[[IN_ADDR_READY:.+]] = firrtl.subfield %[[VAL_1]](1)
// CHECK: %[[IN_ADDR_DATA:.+]] = firrtl.subfield %[[VAL_1]](2)
// CHECK: %[[IN_CONTROL_VALID:.+]] = firrtl.subfield %[[VAL_2]](0)
// CHECK: %[[IN_CONTROL_READY:.+]] = firrtl.subfield %[[VAL_2]](1)
// CHECK: %[[OUT_DATA_VALID:.+]] = firrtl.subfield %[[VAL_3]](0)
// CHECK: %[[OUT_DATA_READY:.+]] = firrtl.subfield %[[VAL_3]](1)
// CHECK: %[[OUT_DATA_DATA:.+]] = firrtl.subfield %[[VAL_3]](2)
// CHECK: %[[OUT_ADDR_VALID:.+]] = firrtl.subfield %[[VAL_4]](0)
// CHECK: %[[OUT_ADDR_READY:.+]] = firrtl.subfield %[[VAL_4]](1)
// CHECK: %[[OUT_ADDR_DATA:.+]] = firrtl.subfield %[[VAL_4]](2)

// CHECK: %[[ALL_VALID_WIRE:inputsValid]] = firrtl.wire : !firrtl.uint<1>

// CHECK: %[[ALL_READY:.+]] = firrtl.and %[[OUT_DATA_READY]], %[[OUT_ADDR_READY]]

// CHECK: %[[ALL_VALID0:.+]] = firrtl.and %[[IN_ADDR_VALID]], %[[IN_DATA_VALID]]
// CHECK: %[[ALL_VALID:.+]] = firrtl.and %[[IN_CONTROL_VALID]], %[[ALL_VALID0]]

// CHECK: firrtl.connect %[[ALL_VALID_WIRE]], %[[ALL_VALID]]

// CHECK: %[[ALL_DONE:.+]] = firrtl.and %[[ALL_READY]], %[[ALL_VALID]]

// CHECK: firrtl.connect %[[IN_DATA_READY]], %[[ALL_DONE]]
// CHECK: firrtl.connect %[[IN_ADDR_READY]], %[[ALL_DONE]]
// CHECK: firrtl.connect %[[IN_CONTROL_READY]], %[[ALL_DONE]]

// CHECK: firrtl.connect %[[OUT_ADDR_DATA]], %[[IN_ADDR_DATA]]
// CHECK: firrtl.connect %[[OUT_DATA_DATA]], %[[IN_DATA_DATA]]

// CHECK: firrtl.connect %[[OUT_DATA_VALID]], %[[ALL_VALID_WIRE]]
// CHECK: firrtl.connect %[[OUT_ADDR_VALID]], %[[ALL_VALID_WIRE]]

// CHECK: firrtl.module @main(in %[[VAL_24:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in %[[VAL_25:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_26:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_27:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out %[[VAL_28:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_29:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_30:.*]]: !firrtl.clock, in %[[VAL_31:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_32:.*]], %[[VAL_33:.*]], %[[VAL_34:.*]], %[[VAL_35:.*]], %[[VAL_36:.*]] = firrtl.instance handshake_store  @handshake_store_in_ui8_ui64_out_ui8_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>, out [[ARG4:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @main(%arg0: i8, %arg1: index, %arg2: none, ...) -> (i8, index, none) {
  %0:2 = "handshake.store"(%arg0, %arg1, %arg2) : (i8, index, none) -> (i8, index)
  handshake.return %0#0, %0#1, %arg2 : i8, index, none
}
