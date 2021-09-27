// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_store_in_ui8_ui64_out_ui8_ui64
// CHECK: %[[IN_DATA_VALID:.+]] = firrtl.subfield %arg0(0)
// CHECK: %[[IN_DATA_READY:.+]] = firrtl.subfield %arg0(1)
// CHECK: %[[IN_DATA_DATA:.+]] = firrtl.subfield %arg0(2)
// CHECK: %[[IN_ADDR_VALID:.+]] = firrtl.subfield %arg1(0)
// CHECK: %[[IN_ADDR_READY:.+]] = firrtl.subfield %arg1(1)
// CHECK: %[[IN_ADDR_DATA:.+]] = firrtl.subfield %arg1(2)
// CHECK: %[[IN_CONTROL_VALID:.+]] = firrtl.subfield %arg2(0)
// CHECK: %[[IN_CONTROL_READY:.+]] = firrtl.subfield %arg2(1)
// CHECK: %[[OUT_DATA_VALID:.+]] = firrtl.subfield %arg3(0)
// CHECK: %[[OUT_DATA_READY:.+]] = firrtl.subfield %arg3(1)
// CHECK: %[[OUT_DATA_DATA:.+]] = firrtl.subfield %arg3(2)
// CHECK: %[[OUT_ADDR_VALID:.+]] = firrtl.subfield %arg4(0)
// CHECK: %[[OUT_ADDR_READY:.+]] = firrtl.subfield %arg4(1)
// CHECK: %[[OUT_ADDR_DATA:.+]] = firrtl.subfield %arg4(2)

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

// CHECK-LABEL: firrtl.module @main
handshake.func @main(%arg0: i8, %arg1: index, %arg2: none, ...) -> (i8, index, none) {
  // CHECK: {{.+}} = firrtl.instance @handshake_store_in_ui8_ui64_out_ui8_ui64
  %0:2 = "handshake.store"(%arg0, %arg1, %arg2) : (i8, index, none) -> (i8, index)

  handshake.return %0#0, %0#1, %arg2 : i8, index, none
}
